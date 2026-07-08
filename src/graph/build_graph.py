"""Assemble the knowledge graph from extracted relationship triples.

Reads every per-company triple file under data/triples/ and merges them into
one directed graph, applying two consolidation rules:

    1. Inverse folding. "A customer_of B" and "B supplier_of A" state the same
       fact, so both are stored as the canonical supplier_of direction. Keeping
       one canonical form halves duplicate edges and makes traversals simpler.
    2. Corroboration merging. The same relationship is often asserted by more
       than one filing (NVIDIA's 10-K says Micron supplies it; Micron's 10-K
       says NVIDIA is a customer). Assertions merge into a single edge that
       records every supporting filing and evidence quote. Edges backed by
       multiple independent documents carry more confidence, which downstream
       retrieval can use.

Symmetric relations (competitor_of, partner_of) are stored once in a
direction-independent canonical order (alphabetical by node name) so that the
pair is a single edge regardless of which side asserted it.

The graph is written to data/graph/knowledge_graph.json in node-link format,
which is human-readable and loads back losslessly with NetworkX.

Run from the project root:
    python -m src.graph.build_graph
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRIPLES_DIR = PROJECT_ROOT / "data" / "triples"
GRAPH_DIR = PROJECT_ROOT / "data" / "graph"
GRAPH_FILE = GRAPH_DIR / "knowledge_graph.json"
COMPANIES_FILE = PROJECT_ROOT / "config" / "companies.json"

# Relations where "A rel B" and "B rel A" mean the same thing. Stored once in
# alphabetical node order.
SYMMETRIC_RELATIONS = {"competitor_of", "partner_of"}

# customer_of is the inverse of supplier_of; every customer_of assertion is
# rewritten to the supplier_of direction on load.
INVERSE_TO_CANONICAL = {"customer_of": "supplier_of"}


def canonical_edge(source: str, relation: str, target: str) -> Tuple[str, str, str]:
    """Return the canonical (source, relation, target) for one assertion."""
    if relation in INVERSE_TO_CANONICAL:
        # A customer_of B  ->  B supplier_of A
        return target, INVERSE_TO_CANONICAL[relation], source
    if relation in SYMMETRIC_RELATIONS and source.lower() > target.lower():
        return target, relation, source
    return source, relation, target


def load_triples() -> List[Dict]:
    """Load every company's extracted triples and re-normalise their names.

    Canonicalisation is re-applied at load time so that improvements to the
    alias table and normalisation rules reach triples extracted earlier
    without re-running the (API-expensive) extraction. Triples whose names no
    longer survive normalisation are dropped here.
    """
    from src.extract.normalize import canonicalize

    triples: List[Dict] = []
    for path in sorted(TRIPLES_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for triple in payload["triples"]:
            source = canonicalize(triple["source"])
            target = canonicalize(triple["target"])
            if source is None or target is None:
                continue
            if source.lower() == target.lower():
                continue
            triple["source"], triple["target"] = source, target
            triples.append(triple)
    return triples


def build_graph(triples: List[Dict]) -> nx.DiGraph:
    """Merge triples into a directed graph with corroboration metadata.

    Each unique (source, relation, target) after canonicalisation becomes one
    edge. The edge accumulates, per supporting assertion: the filing ticker it
    came from, the section, and the evidence quote. num_sources counts the
    distinct filings backing the edge.
    """
    corpus_tickers = {
        c["ticker"]
        for c in json.loads(COMPANIES_FILE.read_text(encoding="utf-8"))["companies"]
    }

    graph = nx.DiGraph()
    # One node per case-insensitive name: the first surface form seen becomes
    # the display name, and later case variants ("Ansys" vs "ANSYS") map onto
    # it instead of creating a second node.
    display_by_lower: Dict[str, str] = {}

    for triple in triples:
        source, relation, target = canonical_edge(
            triple["source"], triple["relation"], triple["target"]
        )
        source = display_by_lower.setdefault(source.lower(), source)
        target = display_by_lower.setdefault(target.lower(), target)

        for node in (source, target):
            if node not in graph:
                graph.add_node(node, in_corpus=False)

        if graph.has_edge(source, target) and graph[source][target]["relation"] == relation:
            edge = graph[source][target]
        elif graph.has_edge(source, target):
            # A different relation already exists between this pair (for
            # example both supplier_of and competitor_of, which is common:
            # Samsung both supplies and competes with NVIDIA). A DiGraph holds
            # one edge per direction, so the second relation is appended to
            # the same edge's relation list.
            edge = graph[source][target]
            if relation not in edge["relations"]:
                edge["relations"].append(relation)
            edge.setdefault("assertions", []).append(
                {
                    "relation": relation,
                    "asserted_by": triple["ticker"],
                    "section": triple.get("section", ""),
                    "evidence": triple.get("evidence", ""),
                }
            )
            edge["num_sources"] = len({a["asserted_by"] for a in edge["assertions"]})
            continue
        else:
            graph.add_edge(
                source,
                target,
                relation=relation,
                relations=[relation],
                assertions=[],
                num_sources=0,
            )
            edge = graph[source][target]

        edge["assertions"].append(
            {
                "relation": relation,
                "asserted_by": triple["ticker"],
                "section": triple.get("section", ""),
                "evidence": triple.get("evidence", ""),
            }
        )
        edge["num_sources"] = len({a["asserted_by"] for a in edge["assertions"]})

    # Mark which nodes are corpus companies (we ingested their filing) versus
    # external entities that only appear inside other companies' filings.
    ticker_names = {
        c["ticker"]: c["name"]
        for c in json.loads(COMPANIES_FILE.read_text(encoding="utf-8"))["companies"]
    }
    # Nodes were extracted under short names; map corpus membership by the
    # short-name forms produced by normalisation.
    from src.extract.normalize import canonicalize

    short_to_ticker = {}
    for ticker, full_name in ticker_names.items():
        short = canonicalize(full_name)
        if short:
            short_to_ticker[short.lower()] = ticker

    for node in graph.nodes:
        ticker = short_to_ticker.get(node.lower())
        if ticker:
            graph.nodes[node]["in_corpus"] = True
            graph.nodes[node]["ticker"] = ticker

    return graph


def save_graph(graph: nx.DiGraph) -> None:
    """Write the graph to disk in node-link JSON format."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(graph, edges="edges")
    GRAPH_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_graph() -> nx.DiGraph:
    """Load the graph previously written by save_graph."""
    data = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="edges", directed=True)


def print_stats(graph: nx.DiGraph) -> None:
    """Print a summary of the assembled graph."""
    corpus_nodes = [n for n, d in graph.nodes(data=True) if d.get("in_corpus")]
    external_nodes = [n for n, d in graph.nodes(data=True) if not d.get("in_corpus")]
    corroborated = [
        (u, v) for u, v, d in graph.edges(data=True) if d.get("num_sources", 0) > 1
    ]

    print(f"Nodes: {graph.number_of_nodes()} "
          f"({len(corpus_nodes)} corpus companies, {len(external_nodes)} external)")
    print(f"Edges: {graph.number_of_edges()} "
          f"({len(corroborated)} corroborated by more than one filing)")

    # Most-connected nodes reveal the hubs of the ecosystem.
    by_degree = sorted(graph.degree, key=lambda item: item[1], reverse=True)
    print("Top hubs by degree:")
    for node, degree in by_degree[:10]:
        marker = "corpus" if graph.nodes[node].get("in_corpus") else "external"
        print(f"  {node:<28} degree={degree:<4} ({marker})")


def main() -> int:
    triples = load_triples()
    if not triples:
        print(f"No triples found in {TRIPLES_DIR}. Run the extractor first.")
        return 1

    print(f"Loaded {len(triples)} triples")
    graph = build_graph(triples)
    save_graph(graph)
    print(f"Graph written to {GRAPH_FILE}")
    print("-" * 60)
    print_stats(graph)
    return 0


if __name__ == "__main__":
    sys.exit(main())

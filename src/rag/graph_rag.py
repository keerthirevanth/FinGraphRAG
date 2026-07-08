"""GraphRAG: answer questions by traversing the knowledge graph.

Where vector retrieval finds text that looks like the question, graph
retrieval follows relationships. The pipeline:

    1. Entity linking. Graph nodes mentioned in the question are found by
       name matching (deterministic, no model call).
    2. Subgraph retrieval. With two or more entities, every simple path
       between them up to a hop limit is collected; the edges along those
       paths are the context. With one entity, its neighbourhood is the
       context. This is what makes multi-hop questions answerable: a path
       Microsoft -> NVIDIA -> TSMC exists in the graph even though no single
       filing states it.
    3. Fact serialisation. Each edge becomes one line carrying the relation,
       which filing asserted it, and the supporting evidence quote, so the
       final answer stays traceable to source documents.
    4. Generation. The model answers strictly from the serialised facts.

Ask a single question from the command line:
    python -m src.rag.graph_rag "How is Microsoft connected to TSMC?"
"""

import itertools
import sys
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from src.graph.build_graph import load_graph
from src.llm_client import LLMClient

# Longest path considered between two entities. Three hops covers the chains
# that matter here (customer -> maker -> supplier) while keeping the fact list
# small enough to stay meaningful.
MAX_HOPS = 3

# Cap on neighbourhood size for single-entity questions, so hub nodes with
# dozens of edges do not flood the context.
MAX_NEIGHBOURHOOD_FACTS = 40

ANSWER_SYSTEM_PROMPT = """You answer questions about relationships between companies using facts from a knowledge graph built from SEC 10-K filings.

Rules:
1. Answer only from the provided facts. If they do not contain the answer, say so plainly.
2. Connections are often indirect. A chain of facts through intermediate companies is a valid connection: if A supplies C and B supplies C, then A and B are connected through their shared customer C. Reason over such chains rather than requiring a single direct fact.
3. When the answer involves a chain, spell it out step by step, naming each relationship.
4. Each fact names the filing it came from; cite it, for example [NVDA 10-K].
5. Be concise and factual. Do not invent relationships that no chain of facts supports."""


class GraphRAG:
    """Graph-traversal retrieval with LLM answer generation."""

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self.graph = load_graph()
        self.client = client or LLMClient()
        # Undirected view used for path finding: a supply relationship
        # connects two companies regardless of direction of travel.
        self._undirected = self.graph.to_undirected(as_view=True)
        # Lowercase name -> node lookup for entity linking.
        self._by_lower = {node.lower(): node for node in self.graph.nodes}

    def link_entities(self, question: str) -> List[str]:
        """Return graph nodes whose names appear in the question.

        Longest names are matched first so that "Micron Technology" wins over
        any shorter overlapping name. Matching is case-insensitive and purely
        lexical: deterministic and free.
        """
        text = question.lower()
        found: List[str] = []
        for lower_name in sorted(self._by_lower, key=len, reverse=True):
            if lower_name in text and self._by_lower[lower_name] not in found:
                found.append(self._by_lower[lower_name])
        return found

    def _edge_fact(self, source: str, target: str) -> str:
        """Serialise one edge into a single traceable fact line."""
        data = self.graph.get_edge_data(source, target)
        if data is None:
            # The traversal ran on the undirected view; recover direction.
            source, target = target, source
            data = self.graph.get_edge_data(source, target)
        relations = ", ".join(data.get("relations", [data.get("relation", "related_to")]))
        assertion = data["assertions"][0] if data.get("assertions") else {}
        asserted_by = assertion.get("asserted_by", "unknown")
        evidence = assertion.get("evidence", "")
        fact = f"{source} --[{relations}]--> {target} (from {asserted_by} 10-K"
        if evidence:
            fact += f'; evidence: "{evidence}"'
        return fact + ")"

    def _paths_between(self, entities: List[str]) -> List[List[str]]:
        """All simple paths up to MAX_HOPS between every pair of entities."""
        paths: List[List[str]] = []
        for source, target in itertools.combinations(entities, 2):
            try:
                found = nx.all_simple_paths(
                    self._undirected, source, target, cutoff=MAX_HOPS
                )
                paths.extend(found)
            except nx.NodeNotFound:
                continue
        return paths

    def retrieve(self, question: str) -> Dict:
        """Collect the graph facts relevant to a question.

        Returns the linked entities, the paths found, and the serialised
        facts. Kept separate from answer() so the evaluation harness and the
        interface can inspect exactly what the model saw.
        """
        entities = self.link_entities(question)
        facts: List[str] = []
        seen_edges: Set[Tuple[str, str]] = set()
        paths: List[List[str]] = []

        if len(entities) >= 2:
            paths = self._paths_between(entities)
            for path in paths:
                for step in range(len(path) - 1):
                    edge = tuple(sorted((path[step], path[step + 1])))
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        facts.append(self._edge_fact(path[step], path[step + 1]))

        # Fall back to (or start from) the neighbourhood of each entity when
        # there are no pairwise paths, or only one entity was mentioned.
        if not facts and entities:
            for entity in entities:
                for neighbour in self._undirected.neighbors(entity):
                    if len(facts) >= MAX_NEIGHBOURHOOD_FACTS:
                        break
                    edge = tuple(sorted((entity, neighbour)))
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        facts.append(self._edge_fact(entity, neighbour))

        return {"entities": entities, "paths": paths, "facts": facts}

    def answer(self, question: str) -> Dict:
        """Answer a question from graph facts; includes the retrieval trace."""
        retrieval = self.retrieve(question)

        if not retrieval["entities"]:
            return {
                "question": question,
                "answer": (
                    "No company mentioned in the question could be matched to "
                    "the knowledge graph."
                ),
                **retrieval,
            }
        if not retrieval["facts"]:
            return {
                "question": question,
                "answer": (
                    "The knowledge graph contains no relationships for the "
                    "companies mentioned within the hop limit."
                ),
                **retrieval,
            }

        facts_block = "\n".join(f"- {fact}" for fact in retrieval["facts"])
        response = self.client.chat(
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Facts:\n{facts_block}\n\nQuestion: {question}",
                },
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return {"question": question, "answer": response, **retrieval}


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m src.rag.graph_rag "your question"')
        return 1
    question = " ".join(sys.argv[1:])
    rag = GraphRAG()
    result = rag.answer(question)

    print("Question:", result["question"])
    print("Linked entities:", ", ".join(result["entities"]) or "none")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)
    print(f"Facts used ({len(result['facts'])}):")
    for fact in result["facts"][:15]:
        print(" ", fact)
    return 0


if __name__ == "__main__":
    sys.exit(main())

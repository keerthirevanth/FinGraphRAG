"""Generate an evaluation question set from the knowledge graph.

The knowledge graph provides ground truth: for each question type the correct
answer is read directly from graph structure, so the set is objective rather
than hand-invented. Every question carries a machine-checkable set of
answer_entities (used for automatic entity-recall scoring) and a readable
reference answer (used by the LLM-judge correctness metric).

Three question types are produced, chosen to separate the two systems:

    simple    - one company's direct relationships, e.g. its competitors.
                The answer sits in that company's own filing, so vector
                retrieval should do well.
    multi_hop - how two non-adjacent corpus companies connect, i.e. their
                shared neighbours. The answer spans several filings and has
                no single supporting passage, which favours graph traversal.
    global    - every corpus company linked to a shared external entity
                (Samsung, TSMC, ...). Answering needs the whole corpus at
                once, which is the graph's home ground.

The generator is deterministic (inputs sorted, fixed seed) so re-running
produces the same set, and it is meant to be reviewed by hand afterwards.

Run from the project root:
    python -m src.eval.generate_questions
"""

import json
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import networkx as nx

from src.graph.build_graph import load_graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.json"

SEED = 20260708

# Targets per type, and thresholds that keep questions substantive.
N_SIMPLE = 30
N_MULTI_HOP = 30
N_GLOBAL = 20
MIN_SIMPLE_ANSWERS = 3      # a company needs at least this many neighbours of
                            # the chosen relation to make a fair question
MIN_GLOBAL_COMPANIES = 3    # an external hub must link at least this many
                            # corpus companies to anchor a global question


def corpus_nodes(graph: nx.DiGraph) -> List[str]:
    """Sorted list of nodes whose own filing we ingested."""
    return sorted(n for n, d in graph.nodes(data=True) if d.get("in_corpus"))


def _relation_neighbours(graph: nx.DiGraph, node: str, relation: str,
                         incoming: bool) -> List[str]:
    """Neighbours joined to node by a specific relation.

    incoming=False looks at edges out of node; incoming=True at edges into it.
    A single edge can carry several relations (stored in 'relations'), so
    membership is tested against that list.
    """
    result = []
    edges = graph.in_edges(node, data=True) if incoming else graph.out_edges(node, data=True)
    for u, v, data in edges:
        if relation in data.get("relations", [data.get("relation")]):
            result.append(u if incoming else v)
    return sorted(set(result))


def make_simple_questions(graph: nx.DiGraph, rng: random.Random) -> List[Dict]:
    """One-company, one-relation questions with direct-neighbour answers."""
    # (relation, incoming, question template, answer-noun) definitions. Because
    # customer_of is folded into supplier_of at graph-build time, supplier and
    # customer questions are both phrased over the supplier_of edge direction.
    specs = [
        ("competitor_of", False, "Who are the main competitors of {c}?", "competitors"),
        ("supplier_of", True, "Which companies are suppliers to {c}?", "suppliers"),
        ("supplier_of", False, "Which companies does {c} supply?", "customers"),
        ("partner_of", False, "Which companies has {c} partnered with?", "partners"),
    ]
    candidates = []
    for company in corpus_nodes(graph):
        for relation, incoming, template, noun in specs:
            answers = _relation_neighbours(graph, company, relation, incoming)
            if len(answers) >= MIN_SIMPLE_ANSWERS:
                candidates.append(
                    {
                        "type": "simple",
                        "question": template.format(c=company),
                        "answer_entities": answers,
                        "reference_answer": (
                            f"According to the filings, the {noun} of {company} "
                            f"include: {', '.join(answers)}."
                        ),
                        "meta": {"focus": company, "relation": relation},
                    }
                )
    rng.shuffle(candidates)
    # Spread across companies: keep at most two questions per focus company so
    # the set is not dominated by the hubs.
    picked, per_company = [], {}
    for q in candidates:
        c = q["meta"]["focus"]
        if per_company.get(c, 0) >= 2:
            continue
        per_company[c] = per_company.get(c, 0) + 1
        picked.append(q)
        if len(picked) >= N_SIMPLE:
            break
    return picked


def make_multi_hop_questions(graph: nx.DiGraph, rng: random.Random) -> List[Dict]:
    """Connection questions between non-adjacent corpus companies.

    The answer is the set of shared neighbours through which the two companies
    connect. Only pairs with no direct edge are used, so a correct answer
    genuinely requires combining information from more than one filing.
    """
    undirected = graph.to_undirected(as_view=True)
    companies = corpus_nodes(graph)

    candidates = []
    for a, b in combinations(companies, 2):
        if undirected.has_edge(a, b):
            continue  # directly connected: not a multi-hop question
        shared = sorted(set(undirected.neighbors(a)) & set(undirected.neighbors(b)))
        if not shared:
            continue
        candidates.append(
            {
                "type": "multi_hop",
                "question": f"How is {a} connected to {b}?",
                "answer_entities": shared,
                "reference_answer": (
                    f"{a} and {b} are connected through shared relationships "
                    f"with: {', '.join(shared)}."
                ),
                "meta": {"entity_a": a, "entity_b": b, "num_connectors": len(shared)},
            }
        )
    # Prefer pairs with a small, specific set of connectors (a crisp answer)
    # over pairs sharing a dozen hubs, then diversify across companies.
    candidates.sort(key=lambda q: q["meta"]["num_connectors"])
    rng.shuffle(candidates)  # break ties randomly but reproducibly
    candidates.sort(key=lambda q: q["meta"]["num_connectors"])

    picked, appearances = [], {}
    for q in candidates:
        a, b = q["meta"]["entity_a"], q["meta"]["entity_b"]
        if appearances.get(a, 0) >= 3 or appearances.get(b, 0) >= 3:
            continue
        appearances[a] = appearances.get(a, 0) + 1
        appearances[b] = appearances.get(b, 0) + 1
        picked.append(q)
        if len(picked) >= N_MULTI_HOP:
            break
    return picked


def make_global_questions(graph: nx.DiGraph, rng: random.Random) -> List[Dict]:
    """Corpus-wide exposure questions anchored on shared external entities."""
    corpus = set(corpus_nodes(graph))
    undirected = graph.to_undirected(as_view=True)

    candidates = []
    for node, data in graph.nodes(data=True):
        if data.get("in_corpus"):
            continue
        linked = sorted(n for n in undirected.neighbors(node) if n in corpus)
        if len(linked) >= MIN_GLOBAL_COMPANIES:
            candidates.append(
                {
                    "type": "global",
                    "question": (
                        f"Which companies in the corpus have a business "
                        f"relationship with {node}?"
                    ),
                    "answer_entities": linked,
                    "reference_answer": (
                        f"The following corpus companies have a disclosed "
                        f"relationship with {node}: {', '.join(linked)}."
                    ),
                    "meta": {"hub": node, "num_companies": len(linked)},
                }
            )
    # Most-shared hubs make the strongest global questions.
    candidates.sort(key=lambda q: q["meta"]["num_companies"], reverse=True)
    return candidates[:N_GLOBAL]


def main() -> int:
    graph = load_graph()
    rng = random.Random(SEED)

    questions = (
        make_simple_questions(graph, rng)
        + make_multi_hop_questions(graph, rng)
        + make_global_questions(graph, rng)
    )
    for index, question in enumerate(questions):
        question["id"] = f"q{index:03d}"

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    QUESTIONS_FILE.write_text(json.dumps(questions, indent=2), encoding="utf-8")

    by_type = {}
    for q in questions:
        by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    print(f"Wrote {len(questions)} questions to {QUESTIONS_FILE}")
    print("By type:", by_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())

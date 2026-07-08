"""Run both RAG systems over the evaluation set and score them.

For every question this harness gets an answer from each system, computes
entity recall against the graph ground truth, and asks the LLM judge for
correctness and faithfulness. Answers and judgements are cached on disk keyed
by their inputs, so an interrupted run resumes for free and re-scoring after a
code change only repeats what actually changed.

Results are written to data/eval/results.json and summarised as a table of
each metric by system and question type.

Run a small check first:
    python -m src.eval.run_eval --limit 6

Run the full evaluation:
    python -m src.eval.run_eval
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.eval.metrics import entity_recall, judge_answer, is_refusal
from src.llm_client import LLMClient
from src.rag.vector_rag import VectorRAG
from src.rag.graph_rag import GraphRAG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.json"
RESULTS_FILE = EVAL_DIR / "results.json"
ANSWER_CACHE = PROJECT_ROOT / "data" / "cache" / "eval_answers"
JUDGE_CACHE = PROJECT_ROOT / "data" / "cache" / "eval_judge"

SYSTEMS = ["vector", "graph"]


def _cache_get(directory: Path, key: str) -> Optional[Dict]:
    path = directory / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _cache_put(directory: Path, key: str, payload: Dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{key}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:24]


def get_answer(system: str, rag, question: str) -> Dict:
    """Return a system's answer and the context it used, via cache.

    The context is captured as text so the faithfulness judge can check the
    answer against exactly what the system saw: retrieved chunks for the
    vector system, serialised graph facts for the graph system.
    """
    key = _hash(system, question)
    cached = _cache_get(ANSWER_CACHE, key)
    if cached is not None:
        return cached

    result = rag.answer(question)
    if system == "vector":
        context = "\n\n".join(
            f"[{c['ticker']} {c['section']}] {c['text']}" for c in result["chunks"]
        )
    else:
        context = "\n".join(f"- {fact}" for fact in result["facts"])

    payload = {"answer": result["answer"], "context": context}
    _cache_put(ANSWER_CACHE, key, payload)
    return payload


def get_judgement(
    judge: LLMClient, system: str, question: str, reference: str,
    answer: str, context: str,
) -> Dict:
    """Return the judge's scores for one answer, via cache."""
    key = _hash("judge", system, question, answer)
    cached = _cache_get(JUDGE_CACHE, key)
    if cached is not None:
        return cached

    scores = judge_answer(judge, question, reference, answer, context)
    _cache_put(JUDGE_CACHE, key, scores)
    return scores


def summarise(results: List[Dict]) -> None:
    """Print average metrics by system and question type."""
    types = ["simple", "multi_hop", "global", "ALL"]
    metrics = ["entity_recall", "correctness", "faithfulness"]

    def average(system: str, qtype: str, metric: str) -> Optional[float]:
        vals = [
            r[metric]
            for r in results
            if r["system"] == system
            and (qtype == "ALL" or r["type"] == qtype)
            and r[metric] is not None
        ]
        return sum(vals) / len(vals) if vals else None

    for metric in metrics:
        print(f"\n=== {metric} ===")
        header = f"{'type':<12}" + "".join(f"{s:>12}" for s in SYSTEMS) + f"{'winner':>10}"
        print(header)
        print("-" * len(header))
        for qtype in types:
            cells = []
            scores = {}
            for system in SYSTEMS:
                value = average(system, qtype, metric)
                scores[system] = value
                cells.append("   n/a" if value is None else f"{value:>12.2f}")
            ranked = [s for s in SYSTEMS if scores[s] is not None]
            winner = max(ranked, key=lambda s: scores[s]) if ranked else "-"
            print(f"{qtype:<12}" + "".join(cells) + f"{winner:>10}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions (for testing)")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    vector_rag = VectorRAG()
    graph_rag = GraphRAG()
    judge = LLMClient()
    rags = {"vector": vector_rag, "graph": graph_rag}

    results: List[Dict] = []
    for index, question in enumerate(questions, start=1):
        q_text = question["question"]
        print(f"[{index}/{len(questions)}] {question['type']:<9} {q_text[:60]}")
        for system in SYSTEMS:
            answered = get_answer(system, rags[system], q_text)
            recall = entity_recall(answered["answer"], question["answer_entities"])
            scores = get_judgement(
                judge, system, q_text, question["reference_answer"],
                answered["answer"], answered["context"],
            )
            # A genuine refusal to an answerable question is incorrect no
            # matter how the judge scored it. A real answer names at least
            # some of the ground-truth entities, so a refusal is only counted
            # when entity recall is zero as well. This distinguishes vector's
            # "the excerpts contain no connection" (recall 0) from a correct
            # graph answer that says "no DIRECT link, but connected through X"
            # (recall > 0, and merely trips the refusal keywords).
            # Faithfulness is left untouched: declining honestly is faithful.
            refused = is_refusal(answered["answer"])
            correctness = scores["correctness"]
            declined = refused and question["answer_entities"] and not recall
            if declined:
                correctness = 0.0
            results.append(
                {
                    "id": question["id"],
                    "type": question["type"],
                    "system": system,
                    "question": q_text,
                    "answer": answered["answer"],
                    "refused": refused,
                    "declined": declined,
                    "entity_recall": recall,
                    "correctness": correctness,
                    "judge_correctness": scores["correctness"],
                    "faithfulness": scores["faithfulness"],
                }
            )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {RESULTS_FILE}")
    summarise(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

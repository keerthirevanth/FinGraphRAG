"""Run the evaluation using only the local self-hosted model.

Experiment-only script (see the experiment/local-35b branch): pins every
answer and judge call to a single local endpoint, taken from position 0 of
LLM_ENDPOINTS, instead of the full cloud pool. This exists to test how a
locally hosted model performs on this project's pipeline, out of curiosity
about local-model usage, not as a replacement for the cloud-based results.

Two things are deliberately isolated from the main evaluation so this run
cannot corrupt or shadow the committed results:

    - Separate cache directories (data/cache/eval_answers_local,
      eval_judge_local). The main cache is keyed by (system, question) only,
      with no model in the key, so sharing it would either silently reuse
      cloud-generated answers or overwrite them with local ones.
    - A separate output file, data/eval/results_local.json, so
      data/eval/results.json (the committed, README-cited results) is
      untouched.

A larger max_tokens is used throughout because this local model produces a
hidden reasoning block before its visible answer; the default budgets tuned
for the cloud pool leave it truncated with no output (see BudgetExhaustedError
notes in llm_client.py for the general pattern this resembles).

Run from the project root:
    python -m src.eval.run_eval_local --limit 10   # quick check
    python -m src.eval.run_eval_local              # full 76 questions
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from src.config import load_llm_config, LLMConfig
from src.eval.metrics import entity_recall, judge_answer, is_refusal
from src.eval.run_eval import summarise
from src.llm_client import LLMClient
from src.rag.vector_rag import VectorRAG
from src.rag.graph_rag import GraphRAG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.json"
RESULTS_FILE = EVAL_DIR / "results_local.json"
ANSWER_CACHE = PROJECT_ROOT / "data" / "cache" / "eval_answers_local"
JUDGE_CACHE = PROJECT_ROOT / "data" / "cache" / "eval_judge_local"

SYSTEMS = ["vector", "graph"]

# This local model reasons silently before answering, so every call needs
# far more headroom than the cloud pool's tuned defaults, confirmed by
# probing: 300-1024 tokens truncated with empty output; 8000 succeeded.
LOCAL_MAX_TOKENS = 4096


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


def get_judgement(judge: LLMClient, system: str, question: str, reference: str,
                   answer: str, context: str) -> Dict:
    key = _hash("judge", system, question, answer)
    cached = _cache_get(JUDGE_CACHE, key)
    if cached is not None:
        return cached
    scores = judge_answer(judge, question, reference, answer, context,
                           max_tokens=LOCAL_MAX_TOKENS)
    _cache_put(JUDGE_CACHE, key, scores)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the eval against the local model only")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    base_config = load_llm_config()
    if not base_config.endpoints:
        print("No endpoints configured.")
        return 1
    local_endpoint = base_config.endpoints[0]
    print(f"Pinned to local endpoint: {local_endpoint.describe()}")
    local_config = LLMConfig(endpoints=(local_endpoint,), timeout=240)

    local_client = LLMClient(local_config)
    vector_rag = VectorRAG(client=local_client, max_tokens=LOCAL_MAX_TOKENS)
    graph_rag = GraphRAG(client=local_client, max_tokens=LOCAL_MAX_TOKENS)
    judge = LLMClient(local_config)
    rags = {"vector": vector_rag, "graph": graph_rag}

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    print(f"Evaluating {len(questions)} questions x {len(SYSTEMS)} systems "
          f"({len(questions) * len(SYSTEMS) * 2} calls if nothing is cached). "
          "This model is slow (~15-40s per call); progress prints as it goes.")

    results = []
    run_start = time.perf_counter()
    for index, question in enumerate(questions, start=1):
        q_text = question["question"]
        for system in SYSTEMS:
            step_start = time.perf_counter()
            answered = get_answer(system, rags[system], q_text)
            recall = entity_recall(answered["answer"], question["answer_entities"])
            scores = get_judgement(
                judge, system, q_text, question["reference_answer"],
                answered["answer"], answered["context"],
            )
            refused = is_refusal(answered["answer"])
            correctness = scores["correctness"]
            declined = refused and question["answer_entities"] and not recall
            if declined:
                correctness = 0.0

            results.append({
                "id": question["id"], "type": question["type"], "system": system,
                "question": q_text, "answer": answered["answer"],
                "refused": refused, "declined": declined,
                "entity_recall": recall, "correctness": correctness,
                "judge_correctness": scores["correctness"],
                "faithfulness": scores["faithfulness"],
            })
            step_elapsed = time.perf_counter() - step_start
            total_elapsed = time.perf_counter() - run_start
            print(f"[{index}/{len(questions)}] {system:<7} {question['type']:<9} "
                  f"{step_elapsed:>5.1f}s (total {total_elapsed/60:.1f}m) "
                  f"recall={recall} corr={correctness}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {RESULTS_FILE} (separate from the committed results.json)")
    summarise(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

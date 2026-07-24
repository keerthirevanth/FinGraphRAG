"""Scoring metrics for the evaluation harness.

Two complementary metrics are used:

    entity_recall - objective and free. Of the ground-truth entities the graph
                    says belong in the answer, how many does the system's
                    answer actually name? Computed by string matching with a
                    small amount of normalisation so that "Micron Technology"
                    in the ground truth still matches "Micron" in an answer.
                    This avoids unfairly rewarding the graph system, whose
                    answers naturally use the exact node names.

    judge scores  - an LLM rates correctness (against the reference answer)
                    and faithfulness (is the answer supported by the context
                    the system retrieved). These capture quality that string
                    matching cannot, at the cost of an API call per judgement.

Keeping both means the headline numbers do not rest on a single, possibly
biased, measure.
"""

import re
from typing import Dict, List, Optional

from src.llm_client import LLMClient


# Phrases that mark an answer as a refusal ("the context does not contain
# this"). Judged against an answerable question, a refusal is incorrect, but
# the LLM judge tends to reward it as a cautious, correct response. Detecting
# refusals deterministically keeps correctness honest and applies the same
# rule to both systems.
_REFUSAL_PATTERNS = [
    "no information", "not mention", "does not mention", "do not mention",
    "no connection", "cannot provide", "cannot answer", "unable to",
    "not contain", "does not contain", "do not contain", "no relevant",
    "there is no", "not provide", "no direct", "insufficient",
    "not enough information", "no mention",
]


def is_refusal(answer: str) -> bool:
    """Whether an answer declines to answer rather than giving content."""
    lowered = answer.lower()
    return any(pattern in lowered for pattern in _REFUSAL_PATTERNS)


def _entity_matches(entity: str, answer_lower: str) -> bool:
    """Whether a ground-truth entity is named in the (lowercased) answer.

    Matches the full name, or a distinctive first word (four or more letters)
    as a whole word. The word-boundary check stops short tokens from matching
    inside unrelated words.
    """
    name = entity.lower()
    if name in answer_lower:
        return True
    first = name.split()[0]
    if len(first) >= 4 and re.search(rf"\b{re.escape(first)}\b", answer_lower):
        return True
    return False


def entity_recall(answer: str, answer_entities: List[str]) -> Optional[float]:
    """Fraction of ground-truth entities named in the answer.

    Returns None when there are no ground-truth entities, so such questions
    can be excluded from recall averages rather than counted as a perfect or
    zero score.
    """
    if not answer_entities:
        return None
    answer_lower = answer.lower()
    hits = sum(1 for entity in answer_entities if _entity_matches(entity, answer_lower))
    return hits / len(answer_entities)


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of question-answering systems for a financial knowledge base built from SEC 10-K filings.

You are given a question, a reference answer (the ground truth), the system's answer, and the context the system was given. Score two things from 0.0 to 1.0:

- "correctness": how well the system's answer matches the facts in the reference answer. 1.0 means it identifies the same companies/relationships; 0.0 means it is wrong or says it cannot answer when the reference has a clear answer. Partial credit for partially correct answers.
- "faithfulness": whether every claim in the system's answer is supported by the provided context. 1.0 means fully supported; 0.0 means it contains claims absent from or contradicting the context. An answer that correctly says the context is insufficient is faithful (1.0).

Respond with only a JSON object, no other text:
{"correctness": <float>, "faithfulness": <float>, "reason": "<one short sentence>"}"""


def judge_answer(
    client: LLMClient,
    question: str,
    reference_answer: str,
    system_answer: str,
    context: str,
    max_tokens: int = 300,
) -> Dict[str, float]:
    """Ask the LLM judge to score correctness and faithfulness.

    On an unparseable judgement the scores default to 0.0 with a note, so a
    malformed judge response is visible rather than silently dropped.
    max_tokens is overridable: some models (reasoning models in particular)
    spend much of the budget on hidden reasoning before the visible JSON.
    """
    user = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference_answer}\n\n"
        f"System answer: {system_answer}\n\n"
        f"Context the system was given:\n{context[:6000]}"
    )
    raw = client.chat(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    parsed = _parse_judge(raw)
    if parsed is None:
        return {"correctness": 0.0, "faithfulness": 0.0, "reason": "unparseable judge response"}
    return parsed


def _parse_judge(raw: str) -> Optional[Dict[str, float]]:
    """Extract the judge's JSON object, tolerating fences and stray text."""
    import json

    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    try:
        return {
            "correctness": max(0.0, min(1.0, float(data["correctness"]))),
            "faithfulness": max(0.0, min(1.0, float(data["faithfulness"]))),
            "reason": str(data.get("reason", ""))[:200],
        }
    except (KeyError, TypeError, ValueError):
        return None

"""Vector RAG baseline: retrieve similar chunks, answer from them.

This is the system GraphRAG has to beat. The pipeline is deliberately
standard: embed the question, take the top-k most similar filing chunks,
and ask the model to answer strictly from those chunks with source citations.
Its strengths and weaknesses are both well known: excellent at questions whose
answer sits inside one passage, weak when the answer requires connecting facts
scattered across different companies' filings, because similarity search has
no notion of relationships.

Ask a single question from the command line:
    python -m src.rag.vector_rag "Who supplies memory to NVIDIA?"
"""

import sys
from typing import Dict, List

from src.llm_client import LLMClient
from src.vector.index import VectorIndex

ANSWER_SYSTEM_PROMPT = """You answer questions about companies using excerpts from their SEC 10-K filings.

Rules:
1. Answer only from the provided excerpts. If they do not contain the answer, say so plainly.
2. Cite the source of each claim using the bracketed excerpt label, for example [NVDA Risk Factors].
3. Be concise and factual."""


def format_context(chunks: List[Dict]) -> str:
    """Render retrieved chunks as labelled excerpts for the prompt."""
    blocks = []
    for chunk in chunks:
        label = f"{chunk['ticker']} {chunk['section']}"
        blocks.append(f"[{label}]\n{chunk['text']}")
    return "\n\n".join(blocks)


class VectorRAG:
    """Retrieve-then-answer pipeline over the vector index."""

    def __init__(
        self, client: LLMClient = None, top_k: int = 5, max_tokens: int = 1024
    ) -> None:
        self.index = VectorIndex()
        self.client = client or LLMClient()
        self.top_k = top_k
        # Some models (reasoning models in particular) spend a large share of
        # the token budget on hidden reasoning before the visible answer, so
        # this is overridable per-instance rather than a fixed constant.
        self.max_tokens = max_tokens

    def retrieve(self, question: str) -> List[Dict]:
        """Return the chunks used as context for a question."""
        return self.index.search(question, top_k=self.top_k)

    def answer(self, question: str) -> Dict:
        """Answer a question; returns the answer text and the chunks used.

        The retrieved chunks are returned alongside the answer so that the
        interface and the evaluation harness can inspect what the model saw.
        """
        chunks = self.retrieve(question)
        response = self.client.chat(
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Excerpts:\n\n{format_context(chunks)}\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return {"question": question, "answer": response, "chunks": chunks}


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m src.rag.vector_rag "your question"')
        return 1
    question = " ".join(sys.argv[1:])
    rag = VectorRAG()
    result = rag.answer(question)

    print("Question:", result["question"])
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)
    print("Retrieved from:")
    for chunk in result["chunks"]:
        print(f"  {chunk['ticker']:<6} {chunk['section']:<14} score={chunk['score']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

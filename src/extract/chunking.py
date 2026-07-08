"""Split long filing sections into chunks suitable for extraction.

Relation extraction works best on focused passages: the model misses far fewer
relationships in a 6,000-character chunk than in a 100,000-character section.
Chunks are cut at paragraph boundaries so that a sentence is never split in
half, which would destroy the relationship it describes.
"""

from typing import List

# Target chunk size in characters. Roughly 1,500 tokens, which leaves ample
# room for the instruction prompt within the model's context and keeps each
# request small enough for free-tier per-minute token limits.
CHUNK_SIZE = 6000

# Chunks shorter than this are merged into their neighbour rather than sent
# to the model alone; a tiny fragment lacks the context to extract from.
MIN_CHUNK_SIZE = 500


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into paragraph-aligned chunks of about chunk_size chars.

    Paragraphs (newline-separated lines from the parser) are accumulated until
    adding the next one would exceed the target size. A paragraph longer than
    the target on its own becomes a chunk by itself rather than being split
    mid-sentence.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        # +1 accounts for the newline that will rejoin the paragraphs.
        if current and current_len + len(paragraph) + 1 > chunk_size:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(paragraph)
        current_len += len(paragraph) + 1

    if current:
        tail = "\n".join(current)
        # Merge a trailing fragment into the previous chunk instead of sending
        # a near-empty request to the model.
        if chunks and len(tail) < MIN_CHUNK_SIZE:
            chunks[-1] = chunks[-1] + "\n" + tail
        else:
            chunks.append(tail)

    return chunks

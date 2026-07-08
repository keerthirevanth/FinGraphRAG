"""Build and query the vector index for the retrieval baseline.

Every parsed filing section is split into retrieval-sized chunks, embedded
with a local sentence-transformers model, and stored as a NumPy matrix plus a
JSONL file of chunk records. Retrieval is exact cosine similarity over the
matrix: at this corpus size (a few thousand chunks) brute-force search returns
in milliseconds, so an approximate-nearest-neighbour index would add a
dependency without adding value.

Retrieval chunks are much smaller than extraction chunks (1,200 vs 6,000
characters). Extraction wants context around each fact; retrieval wants tight
passages so that similarity scores are not diluted by unrelated sentences.

Build the index (one-off, CPU only):
    python -m src.vector.index
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.extract.chunking import chunk_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "data" / "vector_index"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"

# Local embedding model. Small enough for CPU, strong enough for passage
# retrieval; swapping models only requires rebuilding the index.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

RETRIEVAL_CHUNK_SIZE = 1200

SECTION_NAMES = {"1": "Business", "1A": "Risk Factors", "7": "MD&A", "FULL": "Full 10-K"}


def _load_model():
    """Load the embedding model (imported lazily so that querying an already
    built index does not pay the import cost twice)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def build_chunk_records() -> List[Dict]:
    """Split every parsed section into retrieval chunks with provenance."""
    records: List[Dict] = []
    for path in sorted(PROCESSED_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        ticker = payload["ticker"]
        for section_id, text in payload["sections"].items():
            section = SECTION_NAMES.get(section_id, section_id)
            for position, chunk in enumerate(chunk_text(text, RETRIEVAL_CHUNK_SIZE)):
                records.append(
                    {
                        "id": f"{ticker}-{section_id}-{position}",
                        "ticker": ticker,
                        "section": section,
                        "text": chunk,
                    }
                )
    return records


def build_index() -> Tuple[int, int]:
    """Embed all chunks and write the index to disk.

    Returns (number of chunks, embedding dimension). Embeddings are L2
    normalised at build time so that a dot product at query time is cosine
    similarity.
    """
    records = build_chunk_records()
    if not records:
        raise RuntimeError(f"No parsed filings found in {PROCESSED_DIR}")

    model = _load_model()
    texts = [r["text"] for r in records]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    with CHUNKS_FILE.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    return len(records), embeddings.shape[1]


class VectorIndex:
    """Loaded index supporting top-k cosine retrieval."""

    def __init__(self) -> None:
        if not EMBEDDINGS_FILE.exists() or not CHUNKS_FILE.exists():
            raise FileNotFoundError(
                "Vector index not found. Build it with: python -m src.vector.index"
            )
        self.embeddings = np.load(EMBEDDINGS_FILE)
        self.records = [
            json.loads(line)
            for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()
        ]
        self._model = None  # loaded on first query

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return the top_k chunks most similar to the query.

        Each result is the chunk record plus its similarity score. Embeddings
        are pre-normalised, so similarity is a single matrix-vector product.
        """
        if self._model is None:
            self._model = _load_model()
        query_vec = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec
        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for index in order:
            record = dict(self.records[index])
            record["score"] = float(scores[index])
            results.append(record)
        return results


def main() -> int:
    print(f"Building vector index from {PROCESSED_DIR}")
    count, dim = build_index()
    print(f"Indexed {count} chunks (embedding dimension {dim})")
    print(f"Index written to {INDEX_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

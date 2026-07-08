"""Delete triples files that cover only part of their filing.

The extraction run is resumable across days as free-tier budgets refresh, and
--skip-existing skips any ticker whose triples file exists. That is correct
only when the existing file is complete. A file written by an earlier partial
run (for example a --sections 1 pass, or a run cut short mid-company) would
otherwise be treated as done and its remaining sections never extracted.

This script finds those partial files by comparing the chunks_processed count
recorded in each triples file against the number of chunks the filing's full
set of parsed sections produces. A file that processed fewer chunks than the
filing contains is incomplete and is deleted, so the next --skip-existing run
regenerates it (cached chunks are free, only new sections cost quota).

Dry run (report only, delete nothing):
    python -m scripts.prune_partial_triples

Actually delete the partial files:
    python -m scripts.prune_partial_triples --delete
"""

import argparse
import json
import sys
from pathlib import Path

from src.extract.chunking import chunk_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRIPLES_DIR = PROJECT_ROOT / "data" / "triples"


def expected_chunks(ticker: str) -> int:
    """Total chunks across all parsed sections of a ticker's filing."""
    processed = PROCESSED_DIR / f"{ticker}.json"
    if not processed.exists():
        return 0
    sections = json.loads(processed.read_text(encoding="utf-8"))["sections"]
    return sum(len(chunk_text(text)) for text in sections.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune partial triples files")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the partial files (default is a dry-run report).",
    )
    args = parser.parse_args()

    partial = []
    for path in sorted(TRIPLES_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        ticker = payload["ticker"]
        processed = payload.get("chunks_processed", 0)
        expected = expected_chunks(ticker)
        # A small tolerance absorbs off-by-one differences in chunk counting;
        # a genuinely partial file is short by many chunks, not one.
        if expected and processed < expected - 1:
            partial.append((ticker, processed, expected, path))

    if not partial:
        print("No partial triples files found. Everything is complete.")
        return 0

    print(f"{'ticker':<7}{'processed':<11}{'expected':<10}action")
    print("-" * 45)
    for ticker, processed, expected, path in partial:
        action = "deleted" if args.delete else "would delete"
        if args.delete:
            path.unlink()
        print(f"{ticker:<7}{processed:<11}{expected:<10}{action}")

    print("-" * 45)
    if args.delete:
        print(f"Deleted {len(partial)} partial file(s). Re-run the extractor "
              "with --skip-existing to regenerate them.")
    else:
        print(f"{len(partial)} partial file(s) would be deleted. "
              "Re-run with --delete to remove them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

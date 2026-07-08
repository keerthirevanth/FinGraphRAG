"""Parse a downloaded 10-K into clean, section-labelled text.

A 10-K is organised into numbered "Items" (Item 1 Business, Item 1A Risk
Factors, Item 7 MD&A, and so on). The relationship signal we need for the
knowledge graph lives mostly in Items 1, 1A and 7, so the parser isolates
those rather than feeding the whole 200-page document to the model.

Two realities of the actual filings are handled explicitly:
    - Every Item heading appears twice: first in the table of contents, then
      again where the real section begins. The parser locates the body by the
      second occurrence of "Item 1" and segments from there.
    - Headings contain non-breaking spaces and typographic punctuation, which
      are normalised before matching.

Run from the project root to parse every downloaded filing:
    python -m src.ingest.parse_filing
"""

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# The sections that carry business-relationship information. Other items
# (financial statements, controls, etc.) are skipped for extraction.
TARGET_ITEMS = ["1", "1A", "7"]

# Matches a heading line such as "Item 1. Business", "Item 7A. ..." or
# "Item 1A:". Different filers use a period, dash or colon after the item id,
# so all three delimiters are accepted. The item id (e.g. "1", "1A") is group 1.
HEADING_RE = re.compile(r"^item\s+(\d{1,2}[a-z]?)\s*[\.\:\-]", re.IGNORECASE)


def _clean_text(raw: str) -> str:
    """Normalise unicode, collapse whitespace, and drop blank lines.

    Non-breaking spaces and typographic quotes are converted to plain ASCII
    equivalents so that both heading matching and later model input are
    consistent.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("\xa0", " ")
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    """Extract clean plain text from a filing's HTML document."""
    # Imported here so the module loads even in environments that only need the
    # pure-text helpers.
    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    # SEC documents are XHTML; silence the parser's XML-vs-HTML warning since
    # the HTML parser handles them correctly for our purposes.
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(html, "lxml")

    # Modern filings are inline XBRL. The ix:header element holds hidden
    # machine-readable metadata (context dates, us-gaap tags) that would
    # otherwise surface as pages of junk at the start of the text.
    for tag in soup.find_all(["ix:header", "ix:hidden"]):
        tag.decompose()

    return _clean_text(soup.get_text("\n"))


def _find_headings(lines: List[str]) -> List[Tuple[int, str]]:
    """Return (line_index, item_id) for every heading-looking line.

    Only short lines are considered so that a sentence merely mentioning
    "Item 1" in prose is not mistaken for a heading.
    """
    headings = []
    for index, line in enumerate(lines):
        if len(line) > 120:
            continue
        match = HEADING_RE.match(line)
        if match:
            headings.append((index, match.group(1).upper()))
    return headings


def split_items(text: str) -> Dict[str, str]:
    """Segment cleaned filing text into a mapping of item id to section text.

    An item id typically occurs several times: in the table of contents, as
    page-header artifacts repeated through the document, and once as the real
    section heading. Each occurrence produces a candidate segment (its text up
    to the next heading), and the longest candidate wins. TOC entries and page
    headers yield segments a few lines long, while the real section runs for
    pages, so keeping the longest is both simpler and more robust than trying
    to detect where the table of contents ends.
    """
    lines = text.split("\n")
    headings = _find_headings(lines)
    if not headings:
        return {}

    sections: Dict[str, str] = {}
    for position, (line_index, item_id) in enumerate(headings):
        # Candidate section runs from just after this heading to the next
        # heading line of any item.
        next_line = (
            headings[position + 1][0]
            if position + 1 < len(headings)
            else len(lines)
        )
        section_text = "\n".join(lines[line_index + 1:next_line]).strip()
        if len(section_text) > len(sections.get(item_id, "")):
            sections[item_id] = section_text
    return sections


def find_primary_document(ticker: str) -> Optional[Path]:
    """Locate the readable primary document for a ticker's latest 10-K."""
    ticker_dir = RAW_DIR / "sec-edgar-filings" / ticker / "10-K"
    if not ticker_dir.exists():
        return None
    # Each accession is its own folder; pick the most recent by name (accession
    # numbers sort chronologically) and take its primary document.
    accessions = sorted((p for p in ticker_dir.iterdir() if p.is_dir()))
    for accession in reversed(accessions):
        primary = accession / "primary-document.html"
        if primary.exists():
            return primary
    return None


# When the target sections together hold less text than this, item-based
# segmentation has failed (some filers, Intel and Dell among them, put only a
# cross-reference index under the Item headings and organise the body their
# own way). The whole document is then used instead of losing the company.
MIN_SECTION_TEXT = 20000


def parse_ticker(ticker: str) -> Optional[Dict[str, str]]:
    """Parse one ticker's filing and return only the target sections.

    Falls back to the full document text (under the pseudo-section id "FULL")
    for filings where item segmentation produces almost nothing.
    """
    primary = find_primary_document(ticker)
    if primary is None:
        return None
    html = primary.read_text(encoding="utf-8", errors="replace")
    text = html_to_text(html)
    all_sections = split_items(text)
    result = {item: all_sections[item] for item in TARGET_ITEMS if item in all_sections}

    if sum(len(v) for v in result.values()) < MIN_SECTION_TEXT:
        return {"FULL": text}
    return result


def main() -> int:
    companies_file = PROJECT_ROOT / "config" / "companies.json"
    tickers = [c["ticker"] for c in json.loads(companies_file.read_text())["companies"]]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Parsing {len(tickers)} filings into {PROCESSED_DIR}")
    print("-" * 60)

    parsed, missing = 0, []
    for ticker in tickers:
        sections = parse_ticker(ticker)
        if not sections:
            missing.append(ticker)
            print(f"  {ticker:<6} no filing or no target sections")
            continue
        out_path = PROCESSED_DIR / f"{ticker}.json"
        payload = {
            "ticker": ticker,
            "sections": sections,
            "section_lengths": {k: len(v) for k, v in sections.items()},
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        found = ", ".join(f"{k}:{len(v)}" for k, v in sections.items())
        print(f"  {ticker:<6} {found}")
        parsed += 1

    print("-" * 60)
    print(f"Done. Parsed {parsed}, missing {len(missing)}.")
    if missing:
        print("Missing:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())

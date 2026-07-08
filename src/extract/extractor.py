"""Extract typed company relationships from filing text using the LLM.

Each chunk of a filing section is sent to the model with a prompt that forces
a strict JSON output against a fixed relation vocabulary. Free-form extraction
produces inconsistent edge names that fragment the graph, so the model must
choose from the closed set below, and every returned triple is validated
before it is accepted. Invalid or malformed triples are dropped and counted,
never silently kept.

Every accepted triple carries a short evidence quote from the source text.
This makes each edge in the final graph traceable to a sentence in a filing,
which later supports groundedness checks in evaluation.

Run a small pilot first (inspect the output before scaling up):
    python -m src.extract.extractor --tickers NVDA --max-chunks 3

Run the full corpus:
    python -m src.extract.extractor
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.extract import cache
from src.extract.chunking import chunk_text
from src.extract.normalize import canonicalize
from src.llm_client import LLMClient, BudgetExhaustedError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRIPLES_DIR = PROJECT_ROOT / "data" / "triples"

# Bump this whenever the prompt or relation set changes. It is part of the
# cache key, so old cached responses are automatically ignored.
PROMPT_VERSION = "v3"

# Closed vocabulary of relationships. Directions are fixed by definition so
# that the same fact is always stored the same way round.
RELATION_TYPES = {
    "supplier_of": "source supplies components, materials, or services to target",
    "customer_of": "source buys products or services from target",
    "competitor_of": "source and target compete in a market",
    "partner_of": "source and target have a partnership, alliance, or collaboration",
    "subsidiary_of": "source is a subsidiary of target",
    "depends_on": "source relies on target's technology, platform, or manufacturing",
    "invests_in": "source holds or made an investment in target",
}

SYSTEM_PROMPT = """You are an analyst extracting business relationships from SEC 10-K filings.

From the given text, extract relationships between named organizations as JSON.

Rules:
1. Only extract relationships explicitly stated or directly implied in the text.
2. Only use organizations named in the text. Never invent names.
3. The "relation" field must be exactly one of:
   - supplier_of: source supplies components, materials, or services to target
   - customer_of: source buys products or services from target
   - competitor_of: source and target compete in a market
   - partner_of: source and target have a partnership, alliance, or collaboration
   - subsidiary_of: source is a subsidiary of target
   - depends_on: source relies on target's technology, platform, or manufacturing
   - invests_in: source holds or made an investment in target
4. "evidence" must be a short quote (under 200 characters) copied from the text that supports the relationship.
5. Use the company's common short name (for example "NVIDIA" not "NVIDIA Corporation").
6. The filing company is identified in the user message; when the text says "we" or "our", that refers to the filing company.
7. Both source and target must be specific named organizations. Never use generic categories such as "CSPs", "OEMs", "distributors", "customers", "suppliers", "automotive manufacturers", or "third parties". If the text only names a category, extract nothing for it.
8. Get the direction right: if the filing company buys from or relies on X, then X is supplier_of the filing company and the filing company is customer_of X.
9. Ignore biographical text about executives and directors. A person having previously worked at another company is not a business relationship between the companies.
10. If no relationships are present, return an empty list.

Respond with only a JSON array, no other text:
[{"source": "...", "relation": "...", "target": "...", "evidence": "..."}]"""


def build_user_prompt(filing_company: str, section_name: str, chunk: str) -> str:
    """Frame a chunk with the context the model needs to resolve 'we'/'our'."""
    return (
        f"Filing company: {filing_company}\n"
        f"Document: 10-K, section {section_name}\n\n"
        f"Text:\n{chunk}"
    )


def parse_response(raw: str) -> Optional[List[Dict]]:
    """Parse the model's response into a list of dicts, tolerating fences.

    Models sometimes wrap JSON in markdown code fences or add a sentence
    before the array, and reasoning models prepend a <think> block. All of
    that is tolerated; the array is located and parsed. Anything unparseable
    returns None so the caller can count the failure.
    """
    text = raw.strip()
    # Drop a reasoning block if the serving model emitted one.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip a markdown code fence, tolerating trailing chatter after it.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    # First attempt: the text as-is (or from the first bracket onward, when
    # the model wrote a sentence before the array).
    candidates = [text]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    # Salvage attempt for truncated output (token cap reached mid-array):
    # cut back to the last complete object and close the array.
    if start != -1:
        last_obj = text.rfind("}")
        if last_obj > start:
            candidates.append(text[start:last_obj + 1] + "]")
    candidates.append("[]" if text.lstrip().startswith("[]") else None)

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return data

    # Last resort: one malformed element (usually an unescaped quote inside
    # an evidence string) breaks the whole array. Parse each object on its
    # own and keep the well-formed ones rather than losing the entire chunk.
    objects = []
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            item = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            objects.append(item)
    return objects if objects else None


def validate_triples(candidates: List[Dict]) -> List[Dict]:
    """Keep only well-formed triples with a known relation type.

    A triple must have a relation from the closed vocabulary and both entity
    names must survive canonicalisation (generic categories and unusable
    names are rejected there). Source and target must differ after
    normalisation so that alias variants of one company never self-link.
    Evidence is normalised to a string and truncated defensively.
    """
    valid = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        relation = str(item.get("relation", "")).strip()
        evidence = str(item.get("evidence", "")).strip()[:300]
        if relation not in RELATION_TYPES:
            continue
        source = canonicalize(str(item.get("source", "")))
        target = canonicalize(str(item.get("target", "")))
        if source is None or target is None:
            continue
        if source.lower() == target.lower():
            continue
        if BIO_EVIDENCE_RE.search(evidence):
            continue
        valid.append(
            {
                "source": source,
                "relation": relation,
                "target": target,
                "evidence": evidence,
            }
        )
    return valid


def extract_chunk(
    client: LLMClient, filing_company: str, section_name: str, chunk: str
) -> Optional[List[Dict]]:
    """Extract validated triples from one chunk, using the cache when possible."""
    raw = cache.get(PROMPT_VERSION, chunk)
    if raw is None:
        raw = client.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(filing_company, section_name, chunk),
                },
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        served_by = client.config.endpoints[client._current].model
        cache.put(PROMPT_VERSION, chunk, raw, model=served_by)

    parsed = parse_response(raw)
    if parsed is None:
        return None
    return validate_triples(parsed)


# Evidence quotes matching these patterns describe a person's employment
# history, not a relationship between companies. The model occasionally
# relabels executive-bio text as a business relation; the evidence quote
# exposes that, so such triples are rejected here.
BIO_EVIDENCE_RE = re.compile(
    r"employed (at|by)|worked (at|for)|served (as|at)|prior to joining|"
    r"previously (worked|served|held)|joined .* from|designer for|"
    r"director of|officer of|career at",
    re.IGNORECASE,
)


# Human-readable names for the parsed section ids. "FULL" is the parser's
# whole-document fallback for filings that resist item segmentation.
SECTION_NAMES = {
    "1": "Item 1 Business",
    "1A": "Item 1A Risk Factors",
    "7": "Item 7 MD&A",
    "FULL": "Full 10-K",
}


def extract_ticker(
    client: LLMClient,
    ticker: str,
    company_name: str,
    max_chunks: Optional[int] = None,
    only_sections: Optional[List[str]] = None,
) -> Dict:
    """Extract triples from a company's parsed filing sections.

    Returns a summary dict with the triples and counts. max_chunks caps the
    total number of chunks processed, which keeps pilot runs small and cheap.
    only_sections restricts extraction to the given section ids; Item 1 packs
    the most relationships per token by far, so budget-constrained runs do
    it first and add the other sections when quota allows.
    """
    processed_file = PROCESSED_DIR / f"{ticker}.json"
    sections = json.loads(processed_file.read_text(encoding="utf-8"))["sections"]
    if only_sections is not None:
        sections = {k: v for k, v in sections.items() if k in only_sections}

    triples: List[Dict] = []
    seen = set()  # (source, relation, target) keys already recorded
    chunks_done = 0
    parse_failures = 0

    for section_id, section_text in sections.items():
        section_name = SECTION_NAMES.get(section_id, f"Item {section_id}")
        for chunk in chunk_text(section_text):
            if max_chunks is not None and chunks_done >= max_chunks:
                break
            result = extract_chunk(client, company_name, section_name, chunk)
            chunks_done += 1
            if result is None:
                parse_failures += 1
                continue
            # Record provenance on each triple and drop duplicates that were
            # extracted again from another chunk; the first evidence is kept.
            for triple in result:
                key = (
                    triple["source"].lower(),
                    triple["relation"],
                    triple["target"].lower(),
                )
                if key in seen:
                    continue
                seen.add(key)
                triple["ticker"] = ticker
                triple["section"] = section_id
                triples.append(triple)
        if max_chunks is not None and chunks_done >= max_chunks:
            break

    return {
        "ticker": ticker,
        "company": company_name,
        "chunks_processed": chunks_done,
        "parse_failures": parse_failures,
        "triples": triples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract relationship triples")
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Tickers to process (default: all in config/companies.json)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Cap on chunks per company, for cheap pilot runs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip tickers whose triples file already exists. Lets an "
            "interrupted corpus run resume without re-spending API quota, "
            "including after a provider or model switch."
        ),
    )
    parser.add_argument(
        "--sections",
        default=None,
        help=(
            "Comma-separated section ids to extract (e.g. '1' or '1A,7'). "
            "Default is every parsed section. Use '1,FULL' for a first pass "
            "over the densest material when API budget is tight."
        ),
    )
    args = parser.parse_args()
    only_sections = args.sections.split(",") if args.sections else None

    companies_file = PROJECT_ROOT / "config" / "companies.json"
    companies = json.loads(companies_file.read_text(encoding="utf-8"))["companies"]
    name_by_ticker = {c["ticker"]: c["name"] for c in companies}

    tickers = args.tickers or list(name_by_ticker)
    client = LLMClient()

    TRIPLES_DIR.mkdir(parents=True, exist_ok=True)
    total_triples = 0

    for ticker in tickers:
        if ticker not in name_by_ticker:
            print(f"  {ticker:<6} unknown ticker, skipped")
            continue
        if not (PROCESSED_DIR / f"{ticker}.json").exists():
            print(f"  {ticker:<6} no parsed filing, skipped")
            continue
        if args.skip_existing and (TRIPLES_DIR / f"{ticker}.json").exists():
            print(f"  {ticker:<6} triples already exist, skipped")
            continue

        try:
            summary = extract_ticker(
                client,
                ticker,
                name_by_ticker[ticker],
                max_chunks=args.max_chunks,
                only_sections=only_sections,
            )
        except BudgetExhaustedError:
            # Every endpoint hit its daily cap. All completed chunks are in
            # the cache, so stopping here loses nothing: the next run redoes
            # this ticker with the cached chunks free of charge.
            print(f"  {ticker:<6} stopped: all API daily budgets exhausted")
            print(
                "Progress is cached. Re-run the same command after the "
                "daily quotas reset (or add more endpoints to .env)."
            )
            return 2

        # Pilot runs (--max-chunks) are for inspecting extraction behaviour;
        # they process only part of a filing and must never overwrite a real
        # triples file with partial results.
        if args.max_chunks is None:
            out_path = TRIPLES_DIR / f"{ticker}.json"
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        count = len(summary["triples"])
        total_triples += count
        pilot = " (pilot, not written)" if args.max_chunks is not None else ""
        print(
            f"  {ticker:<6} chunks={summary['chunks_processed']} "
            f"triples={count} parse_failures={summary['parse_failures']}{pilot}"
        )

    print("-" * 60)
    print(f"Done. {total_triples} triples written to {TRIPLES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

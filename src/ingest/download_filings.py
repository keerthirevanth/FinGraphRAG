"""Download the latest 10-K filing for each company in the corpus.

Filings come from SEC EDGAR through the sec-edgar-downloader library. SEC's
fair-access policy requires every request to carry a descriptive User-Agent
with a real contact email; the library builds that header from the company
name and email passed to the Downloader, both read from the environment.

Downloaded filings are written under data/raw/ (git-ignored). Each company
gets the single most recent 10-K, both the raw submission and the parsed
human-readable document.

Run from the project root:
    python -m src.ingest.download_filings
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

# Resolve important paths relative to this file so the script works regardless
# of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPANIES_FILE = PROJECT_ROOT / "config" / "companies.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

load_dotenv()


def load_companies():
    """Return the list of company records from the corpus definition file."""
    with open(COMPANIES_FILE, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["companies"]


def download_one(downloader: Downloader, ticker: str) -> int:
    """Download the most recent 10-K for a single ticker.

    Returns the number of filings actually downloaded (0 if none was found).
    download_details=True also stores the readable primary document, which is
    what the parser will consume later.
    """
    return downloader.get("10-K", ticker, limit=1, download_details=True)


def main() -> int:
    email = os.getenv("SEC_EMAIL", "").strip()
    company_name = os.getenv("SEC_COMPANY_NAME", "RAG-Research-Project").strip()

    if not email:
        print(
            "SEC_EMAIL is not set. SEC's fair-access policy requires a contact "
            "email in the request header. Add SEC_EMAIL to your .env file."
        )
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloader = Downloader(company_name, email, str(RAW_DIR))

    companies = load_companies()
    print(f"Downloading latest 10-K for {len(companies)} companies")
    print(f"Destination: {RAW_DIR}")
    print("-" * 60)

    succeeded, failed = [], []
    for company in companies:
        ticker = company["ticker"]
        try:
            count = download_one(downloader, ticker)
            if count > 0:
                succeeded.append(ticker)
                print(f"  {ticker:<6} downloaded")
            else:
                failed.append(ticker)
                print(f"  {ticker:<6} no 10-K found")
        except Exception as error:  # noqa: BLE001 - reported per ticker, run continues
            failed.append(ticker)
            print(f"  {ticker:<6} ERROR: {error}")

    print("-" * 60)
    print(f"Done. {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed tickers:", ", ".join(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())

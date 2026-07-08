"""Normalise extracted entity names so each organization is one graph node.

Extraction returns the same company under several surface forms ("Micron
Technology, Inc.", "Micron"), and sometimes returns generic categories that
are not organizations at all ("OEMs", "distributors"). Left untreated, the
first fragments one company into several disconnected nodes and the second
fills the graph with junk. Both silently destroy the multi-hop paths the
whole project depends on, so normalisation is enforced in code rather than
trusted to the extraction prompt.
"""

import re
from typing import Optional

# Legal-form suffixes stripped from the end of names. Built as one regex,
# applied repeatedly, so compound endings like "Co., Ltd." fall away in
# successive passes. Word boundaries prevent partial-word damage.
_SUFFIX_RE = re.compile(
    r"[\s,]+("
    r"incorporated|corporation|company|limited|holdings|plc|llc|l\.l\.c\.|"
    r"inc\.?|corp\.?|ltd\.?|co\.?|s\.a\.|n\.v\.|ag|se"
    r")\.?\s*$",
    re.IGNORECASE,
)

# Generic categories the model sometimes returns despite instructions.
# These are classes of counterparties, not named organizations, and any
# triple touching one is rejected outright.
_GENERIC_TERMS = {
    "csps", "csp", "cloud service providers", "oems", "oem", "odms", "odm",
    "isvs", "isv", "aibs", "aib", "add-in board manufacturers",
    "system integrators", "global system integrators", "distributors",
    "resellers", "retailers", "suppliers", "customers", "partners",
    "competitors", "subcontractors", "contract manufacturers", "foundries",
    "automotive manufacturers", "tier-1 automotive suppliers",
    "automakers", "startups", "enterprises", "governments",
    "original equipment manufacturers", "original design manufacturers",
    "independent software vendors", "channel partners", "end customers",
    "third parties", "third-party", "vendors", "manufacturers",
}

# Single words that mark a phrase as a category description, not a name.
_CATEGORY_WORDS = {
    "competitors", "competitor", "suppliers", "supplier", "customers",
    "customer", "partners", "partner", "companies", "manufacturers",
    "providers", "vendors", "distributors", "resellers", "startups",
    "entities", "parties", "based",
    # Regulatory, legal and governmental bodies. These are named in Risk
    # Factors as sources of regulation or litigation, not as commercial
    # counterparties, and the extractor sometimes mislabels them as
    # suppliers or partners. They are not business entities and are dropped.
    # (A national government acting as a customer is kept: "government" is
    # deliberately absent here, so "U.S. government" survives.)
    "regulator", "regulators", "regulatory", "authority", "authorities",
    "court", "courts", "commission", "tribunal", "ministry", "parliament",
    "congress", "administration", "agency", "agencies",
}

# Canonical aliases for companies that appear under multiple well-known
# names across filings. Keys are lowercase post-suffix-stripping forms.
_ALIASES = {
    "advanced micro devices": "AMD",
    "alphabet": "Alphabet",
    "google": "Alphabet",
    "google cloud": "Alphabet",
    "google cloud platform": "Alphabet",
    "amazon.com": "Amazon",
    "amazon web services": "Amazon",
    "aws": "Amazon",
    "hon hai precision industry": "Hon Hai (Foxconn)",
    "foxconn": "Hon Hai (Foxconn)",
    "taiwan semiconductor manufacturing": "TSMC",
    "tsmc": "TSMC",
    "international business machines": "IBM",
    "ibm global financing": "IBM",
    "meta platforms": "Meta",
    "facebook": "Meta",
    "sk hynix": "SK Hynix",
    "hewlett packard enterprise": "HPE",
    "hewlett-packard enterprise": "HPE",
    "arm": "Arm",
    # Corporate-form variants of the same company. Without these entries the
    # same organization splits into several graph nodes and multi-hop paths
    # silently break. Note that plain "Hewlett-Packard" is deliberately NOT
    # mapped to HPE: HP Inc. and Hewlett Packard Enterprise are different
    # companies after the 2015 split.
    "nvidia": "NVIDIA",
    "qualcomm": "Qualcomm",
    "qualcomm technologies": "Qualcomm",
    "ansys": "Ansys",
    "samsung": "Samsung",
    "samsung electronics": "Samsung",
    "huawei": "Huawei",
    "huawei technologies": "Huawei",
    "renesas": "Renesas",
    "renesas electronics": "Renesas",
    "infineon": "Infineon",
    "infineon technologies": "Infineon",
    "keysight": "Keysight",
    "keysight technologies": "Keysight",
    "lenovo": "Lenovo",
    "lenovo group": "Lenovo",
    "marvell": "Marvell Technology",
    "marvell technology group": "Marvell Technology",
    "micron": "Micron Technology",
    "cisco": "Cisco Systems",
    "dell": "Dell Technologies",
    "stmicroelectronics nv": "STMicroelectronics",
    "xconn": "XConn Technologies",
    "celestial": "Celestial AI",
    # Subsidiaries and product brands roll up to the parent: for this graph
    # the unit of analysis is the company, and keeping Azure or Cisco Capital
    # as separate nodes only fragments connectivity.
    "microsoft azure": "Microsoft",
    "azure": "Microsoft",
    "cisco capital": "Cisco Systems",
    "cisco systems capital": "Cisco Systems",
    "dell financial services": "Dell Technologies",
    "sony semiconductor manufacturing": "Sony",
    "fujitsu network communications": "Fujitsu",
    "toshiba electronic devices & storage": "Toshiba",
    "symantec enterprise security": "Symantec",
    "openai opco": "OpenAI",
    "oracle ai database": "Oracle",
    "oracle university": "Oracle",
    # Government-as-customer variants collapse to one node.
    "usg": "U.S. government",
    "us government": "U.S. government",
    "u.s. government": "U.S. government",
    "united states government": "U.S. government",
    "federal government": "U.S. government",
    "u.s. federal government": "U.S. government",
}


def _strip_suffixes(name: str) -> str:
    """Remove trailing legal-form suffixes, repeatedly, from a name."""
    result = name.strip().strip(",")
    while True:
        stripped = _SUFFIX_RE.sub("", result).strip().strip(",").strip()
        if stripped == result:
            return result
        result = stripped


def canonicalize(name: str) -> Optional[str]:
    """Return the canonical node name for an extracted entity, or None.

    None means the entity is not a usable organization name (empty after
    cleaning, or a generic category) and the triple should be dropped.
    """
    cleaned = re.sub(r"\s+", " ", name).strip().strip('"').strip()
    if not cleaned:
        return None
    if cleaned.lower() in _GENERIC_TERMS:
        return None

    # Names in the form "Full Name (ACRONYM)" are common in filings, e.g.
    # "Taiwan Semiconductor Manufacturing Company (TSMC)". Resolve the full
    # name first (it usually hits the alias table); fall back to the acronym.
    paren = re.match(r"^(.*?)\s*\(([^)]{1,20})\)$", cleaned)
    if paren:
        result = canonicalize(paren.group(1))
        return result if result is not None else canonicalize(paren.group(2))

    stripped = _strip_suffixes(cleaned)
    if not stripped:
        return None
    if stripped.lower() in _GENERIC_TERMS:
        return None

    alias = _ALIASES.get(stripped.lower())
    if alias:
        return alias

    # Reject residues that are too short or purely descriptive lowercase
    # phrases ("large cloud companies"); real names carry capitalisation.
    if len(stripped) < 2 or stripped == stripped.lower():
        return None

    # Reject long descriptive phrases ("semiconductor suppliers based in
    # China, Europe, and Israel"). Real organization names are short; the
    # generous limits here keep every legitimate corpus name while cutting
    # sentence-like fragments.
    if len(stripped) > 40 or len(stripped.split()) > 6:
        return None

    # Reject names built from category words rather than a proper name
    # ("China-based competitors"). Any single word matching a category term
    # marks the whole phrase as descriptive.
    words = re.split(r"[\s\-]+", stripped.lower())
    if any(word in _CATEGORY_WORDS for word in words):
        return None

    return stripped

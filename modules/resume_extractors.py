import re
from typing import List, Tuple

HEADER_PATTERN = re.compile(r"^(?:PROFESSIONAL\s+EXPERIENCE|WORK\s+EXPERIENCE|EXPERIENCE)\s*$", re.I | re.M)
NEXT_HEADER_PATTERN = re.compile(r"^\s*[A-Z][A-Z &/’'\-]{3,}\s*$", re.M)  # next ALL-CAPS-ish header

def _extract_experience_block(full_text: str) -> str:
    m = HEADER_PATTERN.search(full_text)
    if not m:
        return ""
    start = m.end()
    nxt = NEXT_HEADER_PATTERN.search(full_text, pos=start)
    end = nxt.start() if nxt else len(full_text)
    return full_text[start:end]

def _maybe_company_from_line(line: str) -> str:
    # Heuristic: company usually precedes a comma or date; avoid bullets.
    if line.strip().startswith(("•", "-", "–")):
        return ""
    # Drop role titles like 'Senior ...' at start of line if followed by employer on next line
    # Prefer the first comma-separated segment with letters.
    seg = line.split("  ")[0].split(",")[0].strip()
    # Filter out obvious role words
    if any(w in seg.lower() for w in ["senior ", "manager", "analyst", "engineer", "developer", "officer", "intern"]):
        # Sometimes company is before a hyphen
        parts = re.split(r"[-–|]", line)
        if parts:
            seg = parts[0].split(",")[0].strip()
    # Keep words with capitals/numbers/&
    if len(seg) >= 2 and any(c.isalpha() for c in seg):
        return seg
    return ""

def extract_companies(full_text: str) -> List[str]:
    block = _extract_experience_block(full_text)
    if not block:
        # fallback: scan whole doc for plausible org lines preceding bullets
        block = full_text
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    candidates: List[str] = []
    for i, ln in enumerate(lines):
        # company often on a standalone line; also allow comma/date trails
        comp = _maybe_company_from_line(ln)
        if comp:
            candidates.append(comp)
    # Deduplicate while preserving order; drop over-long obvious headers
    seen = set()
    cleaned = []
    for c in candidates:
        c = re.sub(r"\s{2,}", " ", c)
        if len(c) > 60:  # likely not a company line
            continue
        # trim trailing geo/date
        c = re.sub(r"\b(?:Nepal|USA|India|United States).*$", "", c).strip(" ,-")
        if c and c not in seen and not c.isupper():
            seen.add(c)
            cleaned.append(c)

    # Final pass: keep lines that look like org names (have at least two tokens or contain &/Inc/LLC/Institute/Hospital/Health)
    org_like = []
    for c in cleaned:
        tokens = c.split()
        if len(tokens) >= 2 or any(t in c for t in ["&", "Inc", "LLC", "Ltd", "Institute", "Hospital", "Health", "Centre", "Center", "Technimus", "Zakipoint"]):
            org_like.append(c)
    return org_like[:10]

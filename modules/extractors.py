import re
from typing import List, Tuple, Callable, Dict

# ---------- small utils ----------
SPACE2 = re.compile(r"\s{2,}")
def _clean(s: str) -> str:
    s = s.strip(" ,;|-–—")
    s = SPACE2.sub(" ", s)
    return s

def _dedupe_keep_order(items: List[str], max_len: int = 80, top_k: int = 20) -> List[str]:
    seen = set()
    out = []
    for it in items:
        it = _clean(it)
        if not it or len(it) > max_len:
            continue
        if it not in seen:
            seen.add(it)
            out.append(it)
        if len(out) >= top_k:
            break
    return out

# ---------- generic extractors ----------
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?|\d{3})[-.\s]?\d{3}[-.\s]?\d{4})")
URL_RE   = re.compile(r"\bhttps?://[^\s)>\]]+\b")
DATE_RE  = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{4}\b|\b\d{4}\b", re.I)

def extract_emails(text: str) -> List[str]:
    return _dedupe_keep_order(EMAIL_RE.findall(text), top_k=50)

def extract_phones(text: str) -> List[str]:
    return _dedupe_keep_order(PHONE_RE.findall(text), top_k=50)

def extract_urls(text: str) -> List[str]:
    return _dedupe_keep_order(URL_RE.findall(text), top_k=100)

def extract_dates(text: str) -> List[str]:
    return _dedupe_keep_order(DATE_RE.findall(text), top_k=100)

# ---------- organizations / companies (works broadly) ----------
BULLET = re.compile(r"^\s*(?:[-–•]|\u2022)\s+")
ORG_HINTS = ("Inc", "LLC", "Ltd", "Corp", "Corporation", "Company", "Co.", "University",
             "Institute", "Hospital", "Centre", "Center", "Health", "Group", "Holdings", "&")

ROLE_WORDS = {"intern","analyst","engineer","developer","manager","lead","officer",
              "consultant","associate","director","scientist","specialist","coordinator",
              "architect","administrator","researcher","fellow","assistant","professor"}

def _line_looks_like_company(line: str) -> bool:
    if BULLET.match(line):
        return False
    seg = re.split(r"\s{2,}| - | – | — | \| |, ", line)[0].strip()
    if not any(c.isalpha() for c in seg):
        return False
    if seg.isupper() and "&" not in seg and "." not in seg:
        return False
    first_tokens = seg.lower().split()[:4]
    if any(t in ROLE_WORDS for t in first_tokens):
        return False
    return (len(seg.split()) >= 2) or any(h in seg for h in ORG_HINTS)

def extract_companies(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cands = []
    for ln in lines:
        if _line_looks_like_company(ln):
            seg = re.split(r"\s{2,}| - | – | — | \| |, ", ln)[0].strip()
            cands.append(seg)
    return _dedupe_keep_order(cands, top_k=15)

# ---------- routing ----------
Extractor = Callable[[str], List[str]]
EXTRACTOR_REGISTRY: Dict[str, Extractor] = {
    "email": extract_emails, "emails": extract_emails,
    "phone": extract_phones, "phones": extract_phones,
    "url": extract_urls, "urls": extract_urls, "link": extract_urls, "links": extract_urls,
    "date": extract_dates, "dates": extract_dates,
    "company": extract_companies, "companies": extract_companies,
    "employer": extract_companies, "employers": extract_companies,
}

def pick_extractor(question: str) -> Tuple[Extractor, str] | Tuple[None, None]:
    q = question.lower()
    for key, fn in EXTRACTOR_REGISTRY.items():
        if key in q or f" {key} " in f" {q} ":
            return fn, key
    return None, None

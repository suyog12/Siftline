from typing import Union, IO
import io

def load_pdf_to_text(fobj_or_path: Union[str, IO[bytes]]) -> str:
    """Extract text from a PDF. Tries pdfplumber, falls back to pypdf."""
    try:
        import pdfplumber
        with pdfplumber.open(fobj_or_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass

    try:
        from pypdf import PdfReader
        reader = PdfReader(fobj_or_path)
        pages = [(pg.extract_text() or "") for pg in reader.pages]
        return "\n".join(pages).strip()
    except Exception:
        return ""

def load_any_to_text(filename: str, raw: bytes) -> str:
    """
    Universal loader:
      - .pdf → PDF extract
      - .txt/.md → decode
      - .docx (if python-docx available)
      - else → best-effort utf-8 decode
    """
    name = filename.lower()
    bio = io.BytesIO(raw)

    if name.endswith(".pdf"):
        return load_pdf_to_text(bio)

    if name.endswith(".txt") or name.endswith(".md"):
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return raw.decode("latin-1", errors="ignore")

    if name.endswith(".docx"):
        try:
            import docx  # python-docx
            doc = docx.Document(bio)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception:
            return ""

    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return raw.decode("latin-1", errors="ignore")

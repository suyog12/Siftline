import re
import streamlit as st

def chunk_text_streaming(
    text: str,
    max_chars: int,
    overlap: int,
    max_total_chars: int = 400_000,
    progress: bool = False
):
    """
    Streaming chunker: yields chunks with overlap and enforces a hard cap on total chars.
    IMPORTANT: preserve newlines so headers/sections survive.
    """
    if not text:
        return
    # Collapse spaces but keep newlines; consolidate multiple newlines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()

    n = len(text)
    produced = 0
    i = 0
    prog = st.progress(0) if progress else None
    steps = max(1, n // max(1, max_chars - overlap))

    step = 0
    while i < n and produced < max_total_chars:
        j = min(i + max_chars, n)
        chunk = text[i:j]
        yield chunk
        produced += len(chunk)
        i = j - overlap
        if i <= 0:
            i = j
        step += 1
        if prog:
            prog.progress(min(1.0, step / steps))

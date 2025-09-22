from __future__ import annotations
from typing import List
from transformers import pipeline

class Summarizer:
    """
    Token-safe, hierarchical summarizer for arbitrarily long documents.
    - Splits input into token windows within the model's max length.
    - Summarizes each window.
    - Re-summarizes the partial summaries into a final executive summary.
    - Same mechanism powers a concise "explain" mode.
    """

    def __init__(self, model_name: str):
        self.pipe = pipeline("summarization", model=model_name, tokenizer=model_name)
        self.tokenizer = self.pipe.tokenizer
        self.model = self.pipe.model
        self.max_input_tokens = int(getattr(self.model.config, "max_position_embeddings", 1024))
        self.window_tokens = max(256, self.max_input_tokens - 64)

    # ---------- utilities ----------

    def estimate_tokens(self, text: str) -> int:
        """Return total token count for given text (no truncation)."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, truncation=False))

    @property
    def max_tokens(self) -> int:
        """Public accessor for the model's max input positions."""
        return self.max_input_tokens

    def _split_into_token_windows(self, text: str, overlap_tokens: int = 50, max_windows: int = 8) -> List[str]:
        if not text:
            return []
        input_ids = self.tokenizer.encode(text, truncation=False)
        if len(input_ids) <= self.window_tokens:
            return [text]

        step = max(1, self.window_tokens - overlap_tokens)
        windows: List[str] = []
        i = 0
        while i < len(input_ids) and len(windows) < max_windows:
            chunk_ids = input_ids[i : i + self.window_tokens]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            windows.append(chunk_text)
            i += step
        return windows

    def _summarize_chunk(self, text: str, max_len: int, min_len: int) -> str:
        out = self.pipe(text, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"].strip()
        return out

    # ---------- public API ----------

    def summarize(self, text: str, max_len: int = 160, min_len: int = 80) -> str:
        if not text:
            return "**Executive Summary**\n\n"

        windows = self._split_into_token_windows(text, overlap_tokens=50, max_windows=8)
        partials = [self._summarize_chunk(w, max_len=max_len, min_len=min_len) for w in windows]

        if len(partials) == 1:
            return f"**Executive Summary**\n\n{partials[0]}"

        combined = " ".join(partials)
        final = self._summarize_chunk(combined, max_len=max_len, min_len=max(1, min_len // 2))
        return f"**Executive Summary**\n\n{final}"

    def explain(self, text: str, question: str, max_len: int = 180, min_len: int = 70) -> str:
        if not text:
            return "**Explanation**\n\n"

        windows = self._split_into_token_windows(text, overlap_tokens=50, max_windows=8)
        prompt_head = f"{question}\n\n"
        partials = [self._summarize_chunk(prompt_head + w, max_len=max_len, min_len=min_len) for w in windows]

        if len(partials) == 1:
            return f"**Explanation**\n\n{partials[0]}"

        combined = " ".join(partials)
        final = self._summarize_chunk(prompt_head + combined, max_len=max_len, min_len=max(1, min_len // 2))
        return f"**Explanation**\n\n{final}"

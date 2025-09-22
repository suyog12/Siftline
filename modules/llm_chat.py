from typing import List, Tuple
from transformers import pipeline

SYS_PROMPT = (
    "You are a precise assistant. Use ONLY the provided context.\n"
    "- If the information is not present, reply exactly: 'Not found in the document.'\n"
    "- Do not guess. Do not use outside knowledge.\n"
    "- If the question asks to name or list entities (companies, emails, phone numbers, URLs, dates, etc.), "
    "output ONLY a semicolon-separated list of those entities, with no extra words.\n"
    "- Otherwise, answer concisely (1–3 sentences)."
)

class QAGenerator:
    def __init__(self, model_name: str):
        self.pipe = pipeline("text2text-generation", model=model_name, tokenizer=model_name)

    def _build_prompt(self, query: str, contexts: List[str], history: List[Tuple[str, str]]):
        hist = ""
        for q, a in history[-3:]:
            hist += f"\nQ: {q}\nA: {a}\n"
        ctx = "\n\n".join(contexts[:4])
        return (
            f"{SYS_PROMPT}\n"
            f"{'Conversation so far:' if hist else ''}{hist}\n"
            f"Context:\n{ctx}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

    def _trim(self, text: str, max_sentences: int = 3) -> str:
        s = text.strip().split("\n")[0]
        parts = [p.strip() for p in s.split(".") if p.strip()]
        return (". ".join(parts[:max_sentences]) + ("." if parts else "")).strip() or s

    def answer(self, query: str, contexts: List[str], history: List[Tuple[str, str]]):
        if not contexts:
            return "Not found in the document."
        prompt = self._build_prompt(query, contexts, history)
        out = self.pipe(prompt, max_new_tokens=128, do_sample=False)[0].get("generated_text", "").strip()
        return self._trim(out, 3) if out else "Not found in the document."

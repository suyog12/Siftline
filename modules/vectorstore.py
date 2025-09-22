from dataclasses import dataclass
from typing import List
import numpy as np
import faiss
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class QueryHits:
    indices: List[int]
    scores: List[float]

def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32")
    rng = float(arr.max() - arr.min())
    return (arr - arr.min()) / (rng + 1e-6)

class InMemoryVectorStore:
    """
    Hybrid retriever with robust recall and speed:
      1) Embedding search (over-retrieve).
      2) TF-IDF top-M on the full corpus.
      3) Union candidates, fuse scores.
    """
    def __init__(self, dim: int, texts: List[str]):
        self.texts = texts
        self._count = 0
        self.index = faiss.IndexFlatIP(dim)
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_mat = self.tfidf.fit_transform(texts)

    @classmethod
    def from_texts_batched(cls, texts: List[str], embedder, batch_size: int = 64, progress: bool = False):
        assert len(texts) > 0, "No texts provided to index."
        first = texts[: min(batch_size, len(texts))]
        emb0 = embedder.encode(first).astype("float32")
        store = cls(emb0.shape[1], texts)
        store.index.add(emb0); store._count += emb0.shape[0]

        total, done = len(texts), len(first)
        prog = st.progress(0) if progress else None
        while done < total:
            batch = texts[done: done + batch_size]
            emb = embedder.encode(batch).astype("float32")
            store.index.add(emb)
            store._count += emb.shape[0]
            done += len(batch)
            if prog:
                prog.progress(done / total)
        if prog:
            prog.progress(1.0)
        return store

    def query(self, query_text: str, embedder, k: int = 6) -> QueryHits:
        if self._count == 0:
            return QueryHits(indices=[], scores=[])

        # 1) Embedding search
        q_emb = embedder.encode([query_text]).astype("float32")
        over_k = min(max(k * 6, k), self._count)
        emb_scores, emb_idx = self.index.search(q_emb, over_k)
        emb_scores = emb_scores[0]; emb_idx = emb_idx[0]

        # 2) TF-IDF top-M across full corpus
        q_vec = self.tfidf.transform([query_text])
        kw_scores_all = cosine_similarity(q_vec, self.tfidf_mat)[0]
        M = min(max(k * 6, k), self._count)
        kw_idx_sorted = np.argsort(-kw_scores_all)[:M]

        # 3) Union candidates + fusion
        union_idx = np.unique(np.concatenate([emb_idx, kw_idx_sorted]))
        emb_map = {int(i): float(s) for i, s in zip(emb_idx.tolist(), emb_scores.tolist())}
        emb_scores_u = np.array([emb_map.get(int(i), 0.0) for i in union_idx], dtype="float32")
        kw_scores_u  = np.array([kw_scores_all[int(i)] for i in union_idx], dtype="float32")

        emb_n = _normalize(emb_scores_u)
        kw_n  = _normalize(kw_scores_u)
        fused = 0.70 * emb_n + 0.30 * kw_n

        order = np.argsort(-fused)[:k]
        final_idx = union_idx[order]
        final_scores = fused[order].tolist()
        return QueryHits(indices=final_idx.tolist(), scores=final_scores)

from typing import List, Tuple

class Reranker:
    """
    Thin wrapper around a cross-encoder for passage reranking.
    If sentence-transformers or the model is unavailable, instantiation will raise,
    and the app will gracefully fall back to the original order.
    """
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder  # lazy import
        except Exception as e:
            raise RuntimeError("CrossEncoder unavailable. Install sentence-transformers to enable reranking.") from e
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, passages: List[Tuple[int, str]], top_k: int = 6) -> List[Tuple[int, float]]:
        pairs = [(query, p) for _, p in passages]
        scores = self.model.predict(pairs).tolist()
        ranked = sorted([(passages[i][0], scores[i]) for i in range(len(passages))], key=lambda x: -x[1])
        return ranked[:top_k]

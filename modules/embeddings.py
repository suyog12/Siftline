from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        # Always return normalized embeddings for cosine/IP compatibility
        return self.model.encode(texts, normalize_embeddings=True)

def get_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)

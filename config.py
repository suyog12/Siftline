# Privacy-first defaults
DEMO_MODE = False
PRIVACY_MODE = True

# RAG / chunking
TOP_K = 6
MAX_CHARS_PER_CHUNK = 900
CHUNK_OVERLAP_CHARS = 120

# Models (balanced for quality + speed)
MODEL_EMBED = "sentence-transformers/all-MiniLM-L6-v2"  # or "BAAI/bge-small-en-v1.5"
MODEL_QA    = "google/flan-t5-base"
MODEL_SUM   = "sshleifer/distilbart-cnn-12-6"

# Optional cross-encoder reranker (set to None to disable)
MODEL_RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
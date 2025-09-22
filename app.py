import os
import glob
import streamlit as st

# Quiet HF console noise but keep in-app warnings we show
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

from config import (
    DEMO_MODE, PRIVACY_MODE,
    MAX_CHARS_PER_CHUNK, CHUNK_OVERLAP_CHARS, TOP_K,
    MODEL_EMBED, MODEL_QA, MODEL_SUM, MODEL_RERANK,
)

from modules.ingestion import load_any_to_text
from modules.embeddings import get_embedder
from modules.vectorstore import InMemoryVectorStore
from modules.llm_chat import QAGenerator
from modules.summarizer import Summarizer
from modules.topics import extract_topics
from utils.helpers import chunk_text_streaming

# Optional local cap for very large files
MAX_TOTAL_CHARS_LOCAL = 400_000

# ---------- Page config & styling ----------
st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
      h1, h2, h3, h4 { font-weight: 600; }
      .stProgress > div > div > div > div { background-color: #0d6efd; }
      .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
      .stTabs [data-baseweb="tab"] {
        background-color: #f5f5f5;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
        font-weight: 600;
      }
      .stTabs [aria-selected="true"] { background-color: #ffffff; }
      .small-muted { color: #666; font-size: 0.92rem; }
      .caption-muted { color: #777; }
      .markdown-text-container p { line-height: 1.45; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI Knowledge Assistant")
st.markdown(
    '<div class="small-muted">Stages: 1) Load models → 2) Read & chunk → 3) Embed → 4) Retrieve → 5) Generate</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="caption-muted">Uploads are processed in memory for this session only. '
    'Models are pretrained/frozen (no fine-tuning/training). Close or refresh to clear.</div>',
    unsafe_allow_html=True
)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Settings")
    st.write(f"Demo mode: **{DEMO_MODE}**")
    st.write(f"Privacy mode: **{PRIVACY_MODE}**")
    st.write(f"Top-k passages: {TOP_K}")
    st.write(f"Chunk size: {MAX_CHARS_PER_CHUNK} chars")
    st.write(f"Overlap: {CHUNK_OVERLAP_CHARS} chars")
    st.write(f"Max total chars (cap): {MAX_TOTAL_CHARS_LOCAL:,}")

    if st.button("Clear Session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ---------- Heavy resources (cached once per process) ----------
@st.cache_resource(show_spinner=False)
def _embedder():
    return get_embedder(MODEL_EMBED)

@st.cache_resource(show_spinner=False)
def _qa():
    return QAGenerator(model_name=MODEL_QA)

@st.cache_resource(show_spinner=False)
def _summarizer():
    return Summarizer(model_name=MODEL_SUM)

with st.status("Initializing models...", expanded=False) as status:
    embedder = _embedder()
    status.update(label="Embedding model loaded", state="running")
    qa = _qa()
    status.update(label="Q&A model loaded", state="running")
    summarizer = _summarizer()
    status.update(label="Summarizer ready", state="complete")

# Warn if using the HF fallback embedder (no sentence-transformers)
try:
    if getattr(embedder, "backend", "") == "hf_transformers_fallback":
        st.warning(
            "sentence-transformers is not installed. Using a Transformers fallback for embeddings. "
            "For best performance, install: pip install -U sentence-transformers"
        )
except Exception:
    pass

# ---------- Session state ----------
ss = st.session_state
ss.setdefault("history", [])         # list[(q, a)]
ss.setdefault("vectorstore", None)
ss.setdefault("chunks", [])
ss.setdefault("doc_name", None)
ss.setdefault("reranker", None)      # optional singleton

# ---------- Document selection / upload ----------
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    if DEMO_MODE:
        st.subheader("Select a sample document (/data)")
        sample_paths = sorted(glob.glob(os.path.join("data", "*.*")))
        if not sample_paths:
            st.info("Place a couple of public PDFs/TXT/DOCX under data/ to try the demo.")
        else:
            choice = st.selectbox("Sample files", sample_paths, index=0)
            if st.button("Load Sample"):
                with st.status("Reading & indexing document...", expanded=True) as status:
                    with open(choice, "rb") as f:
                        raw = f.read()
                    text = load_any_to_text(os.path.basename(choice), raw)

                    status.update(label="Cleaning & chunking (streaming, memory-capped)...", state="running")
                    chunks = list(
                        chunk_text_streaming(
                            text,
                            max_chars=MAX_CHARS_PER_CHUNK,
                            overlap=CHUNK_OVERLAP_CHARS,
                            max_total_chars=MAX_TOTAL_CHARS_LOCAL,
                            progress=True,
                        )
                    )

                    status.update(label="Embedding chunks (batched)...", state="running")
                    store = InMemoryVectorStore.from_texts_batched(
                        chunks, embedder, batch_size=64, progress=True
                    )

                    ss.chunks = chunks
                    ss.vectorstore = store
                    ss.history = []
                    ss.doc_name = os.path.basename(choice)

                    status.update(label="Document ready", state="complete")
                    st.success(f"Loaded: {ss.doc_name} (chunks: {len(chunks):,})")
    else:
        st.subheader("Upload a document (PDF, TXT/MD, DOCX)")
        uploaded = st.file_uploader("Choose a file", type=["pdf", "txt", "md", "docx"])
        if uploaded is not None and st.button("Process Document"):
            with st.status("Reading & indexing document...", expanded=True) as status:
                raw = uploaded.read()
                text = load_any_to_text(getattr(uploaded, "name", "uploaded.bin"), raw)

                status.update(label="Cleaning & chunking (streaming, memory-capped)...", state="running")
                chunks = list(
                    chunk_text_streaming(
                        text,
                        max_chars=MAX_CHARS_PER_CHUNK,
                        overlap=CHUNK_OVERLAP_CHARS,
                        max_total_chars=MAX_TOTAL_CHARS_LOCAL,
                        progress=True,
                    )
                )

                status.update(label="Embedding chunks (batched)...", state="running")
                store = InMemoryVectorStore.from_texts_batched(
                    chunks, embedder, batch_size=64, progress=True
                )

                ss.chunks = chunks
                ss.vectorstore = store
                ss.history = []
                ss.doc_name = getattr(uploaded, "name", "uploaded.bin")

                status.update(label="Document ready", state="complete")
                st.success(f"Processed: {ss.doc_name} (chunks: {len(chunks):,})")

with col_right:
    st.subheader("Document Status")
    if ss.vectorstore is None:
        st.info("No document loaded yet.")
    else:
        st.write("Document:", ss.doc_name or "Loaded")
        st.write("Chunks:", f"{len(ss.chunks):,}")
        st.success("Vector index: ready")

st.markdown("---")

# ---------- Tabs ----------
tab_chat, tab_sum, tab_topics = st.tabs(["Chat with Document", "Summarize Document", "Extract Topics"])

# ---------------- Chat ----------------
with tab_chat:
    st.subheader("Ask questions grounded in the current document")
    if ss.vectorstore is None:
        st.info("Load or process a document first.")
    else:
        q = st.text_input("Your question")
        go = st.button("Answer")

        if go and q.strip():
            question = q.strip()

            # Prepare full text for summarizer/extractors (safe due to hierarchical summarizer)
            full_text = "\n".join(ss.chunks)

            # 1) Summary/Explain intent routing (BEFORE retrieval)
            q_lower = question.lower()
            wants_summary = any(kw in q_lower for kw in ["summary", "summarize", "executive summary"])
            wants_explain = any(kw in q_lower for kw in ["explain", "explanation", "why", "how"])

            if wants_summary or wants_explain:
                # User-visible token size warning
                tok_count = summarizer.estimate_tokens(full_text)
                if tok_count > summarizer.max_tokens:
                    st.warning(
                        f"Long input detected: {tok_count} tokens exceed the model limit "
                        f"({summarizer.max_tokens}). The document will be processed in windows to avoid errors."
                    )
                try:
                    if wants_summary:
                        result = summarizer.summarize(full_text, max_len=160, min_len=80)
                    else:
                        result = summarizer.explain(full_text, question=question, max_len=180, min_len=70)
                except Exception as e:
                    st.error(f"Summarization failed: {e}")
                    st.stop()

                st.write("Answer")
                st.write(result)
                ss.history.append((question, result))
                st.stop()

            # 2) Generic entity extractor routing (works for any document)
            from modules.extractors import pick_extractor
            extractor, key = pick_extractor(question)
            if extractor:
                items = extractor(full_text)

                st.write("Answer")
                if items:
                    st.write("; ".join(items))  # semicolon-separated list only
                else:
                    st.write("Not found in the document.")

                # Transparent sources
                with st.expander("Thinking process (sources)"):
                    hits = ss.vectorstore.query(f"{key} {question}", embedder, k=TOP_K)
                    for i, score in zip(hits.indices, hits.scores):
                        snippet = ss.chunks[i][:400].replace("\n", " ")
                        st.markdown(f"- Chunk #{i} (score={score:.4f})\n\n> {snippet}…")

                ss.history.append((question, "; ".join(items) if items else "Not found in the document."))
                st.stop()

            # 3) Default: Hybrid RAG QA with strict, concise output
            with st.status("Retrieving relevant passages...", expanded=False) as status:
                hits = ss.vectorstore.query(question, embedder, k=TOP_K)

                # Optional reranker: AFTER retrieval, BEFORE answering
                used_indices = list(hits.indices)  # default order
                if MODEL_RERANK:
                    try:
                        from modules.rerank import Reranker
                        if ss.reranker is None:
                            ss.reranker = Reranker(model_name=MODEL_RERANK)
                        passages = [(i, ss.chunks[i]) for i in used_indices]
                        ranked = ss.reranker.rerank(question, passages, top_k=len(passages))
                        used_indices = [int(i) for i, _ in ranked]
                    except Exception:
                        pass

                contexts = [ss.chunks[i] for i in used_indices]

                status.update(label="Generating answer...", state="running")
                answer = qa.answer(query=question, contexts=contexts, history=ss.history)
                status.update(label="Done", state="complete")

            ss.history.append((question, answer))
            st.write("Answer")
            st.write(answer)

            with st.expander("Thinking process (sources)"):
                for i in used_indices:
                    snippet = ss.chunks[i][:400].replace("\n", " ")
                    st.markdown(f"- Chunk #{i}\n\n> {snippet}…")

            if ss.history:
                with st.expander("Conversation so far"):
                    for i, (qq, aa) in enumerate(ss.history, 1):
                        st.markdown(f"**Q{i}:** {qq}\n\n**A{i}:** {aa}\n")

# ---------------- Summarize ----------------
with tab_sum:
    st.subheader("One-click Executive Summary")
    if ss.vectorstore is None:
        st.info("Load or process a document first.")
    else:
        if st.button("Generate Summary"):
            with st.status("Summarizing...", expanded=False) as status:
                full_text = " ".join(ss.chunks)
                tok_count = summarizer.estimate_tokens(full_text)
                if tok_count > summarizer.max_tokens:
                    st.warning(
                        f"Long input detected: {tok_count} tokens exceed the model limit "
                        f"({summarizer.max_tokens}). The document will be processed in windows to avoid errors."
                    )
                try:
                    summary = summarizer.summarize(full_text, max_len=160, min_len=80)
                except Exception as e:
                    st.error(f"Summarization failed: {e}")
                    st.stop()
                status.update(label="Summary ready", state="complete")
            st.write(summary)

# ---------------- Topics ----------------
with tab_topics:
    st.subheader("Unsupervised topic discovery")
    if ss.vectorstore is None:
        st.info("Load or process a document first.")
    else:
        n_topics = st.slider("Number of topics", 2, 8, 4, 1)
        if st.button("Extract Topics"):
            with st.status("Extracting topics...", expanded=False) as status:
                topics = extract_topics(ss.chunks, n_topics=n_topics)
                status.update(label="Topics ready", state="complete")
            for i, words in enumerate(topics, 1):
                st.markdown(f"**Topic {i}:** {', '.join(words)}")

# ---------- Footer ----------
st.markdown("---")
st.caption(
    "Privacy: uploads are processed in memory and are not stored; models are frozen (no training). "
    "For sensitive use, run locally. For public demos, use non-sensitive sample documents."
)

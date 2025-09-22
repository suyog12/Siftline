# Siftline
Private, grounded Q&A and summarization for your documents (PDF, DOCX, TXT/MD).  
Answers are concise, sources are transparent, and your files are processed in-memory only.

---

## Table of Contents
- [What Siftline Does](#what-siftline-does)
- [Why It’s Different](#why-its-different)
- [Current Limitations](#current-limitations)
- [Roadmap / Upgrades Planned](#roadmap--upgrades-planned)
- [Project Status](#project-status)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Run Locally](#run-locally)
- [Deploy on Streamlit Cloud](#deploy-on-streamlit-cloud)
- [Tips & Troubleshooting](#tips--troubleshooting)
- [Repo Structure](#repo-structure)
- [License](#license)

---

## What Siftline Does
Siftline lets you upload a document and ask precise, grounded questions. It uses a hybrid retrieval pipeline and a strict answerer so you get **only what’s in the file**:

- **Multi-format ingestion**: PDF, DOCX, TXT/Markdown.
- **Streaming chunking**: memory-capped, overlap-aware segmentation for small and very large files.
- **Hybrid retrieval**: dense (embeddings via FAISS) + sparse (TF-IDF) with score fusion.
- **Optional re-ranking**: cross-encoder reorders candidates for better precision (if installed).
- **Strict QA**: 
  - “Entity questions” (e.g., *Name the companies I have worked in*) return a **semicolon-separated list** only.
  - Otherwise, answers are **1–3 sentences**, grounded only in retrieved context.
  - If not found, it says exactly: **“Not found in the document.”**
- **Token-safe summarization**: hierarchical (map-reduce) summarizer that splits long inputs to avoid model limits and shows a user-visible warning when your document exceeds the model’s max tokens.
- **Transparent sources**: a “Thinking process (sources)” panel shows the chunk snippets used to answer (no hidden chain-of-thought, just evidence).

---

## Why It’s Different
- **Privacy**: No training, no external calls; files are processed in memory during the session.
- **Grounded**: The QA prompt forbids guessing; outputs are clipped to the retrieved context.
- **Resilient on large PDFs**: Summarizer never feeds an overlong sequence to the model.

---

## Current Limitations
This project uses **free, small-to-medium open models** by default. That brings tradeoffs:

- **Quality vs. size**: `flan-t5-base` (QA) and `distilbart-cnn` (summarization) are solid but can still:
  - Miss subtle references if retrieval doesn’t catch them.
  - Be conservative on edge queries (better than hallucinating).
- **Latency**: First run downloads models; CPU-only inference is slower than GPU.
- **No OCR for scanned PDFs** (yet). Text extraction expects selectable text.
- **No long-term storage**: Sessions are ephemeral by design.
- **No evaluation suite**: There isn’t a benchmark harness included yet.

> You can upgrade models later for higher fidelity and speed (see [Roadmap](#roadmap--upgrades-planned)).

---

## Roadmap / Upgrades Planned
- **Model upgrades (toggleable)**:
  - Better embeddings (e.g., `bge-small-en-v1.5`).
  - Larger instruction-following models when resources allow.
- **Reranker by default** (when `sentence-transformers` is available).
- **OCR fallback** for scanned PDFs (Tesseract).
- **Smarter chunking** (header-aware; page/section boundaries).
- **Structured outputs** option (JSON for entities, tables).
- **Page-aware citations** (page numbers + source preview).
- **Caching** (on-disk model + embedding cache for faster reloads).

---

## Project Status
**Under active development.** Feature set is stable for local/Cloud demos, but expect frequent improvements and occasional breaking changes.

---

## Quickstart
```bash
# Clone
git clone https://github.com/suyog12/Siftline.git
cd Siftline

# (Recommended) Use Python 3.11
# If using conda:
# conda create -n siftline python=3.11 -y && conda activate siftline

# Install deps
pip install -r requirements.txt

# Run
streamlit run app.py

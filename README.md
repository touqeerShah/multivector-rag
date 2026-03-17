# Multivector RAG (Retrieval-Augmented Generation)

## ✅ Project Overview
This repository implements a **hybrid multi-vector retrieval pipeline** for PDF documents, combining:

- **Text retrieval** (BM25 + dense embeddings)
- **Visual retrieval** (page images + multivector embeddings)
- **Reranking** (ColBERT for text, ColQwen2 for page images)
- **Answer generation** with citations and metrics

The end goal is a system that can answer queries using both **textual content** and **visual layout/context** from PDFs.

---

## 🔍 Architecture (End-to-End Flow)

1. **User query**
2. **LangGraph router** selects retrieval paths
3. **Stage 1 – Candidate retrieval**
   - BM25 over extracted text chunks
   - Dense text retrieval
   - MUVERA/FDE proxy retrieval for multi-vector page embeddings
4. **Stage 2 – Reranking**
   - ColBERT for text chunk reranking
   - ColQwen2 for page-image candidate reranking
5. **Context packer** (merges retrieval results)
6. **Answer generation** (LLM prompt with context)
7. **Citations + metrics**

---

## 🧱 PDF Data Model (Two Views)

To support hybrid retrieval, PDFs are stored in two parallel views:

### 1) Text View
- Extracted text
- OCR output
- Chunked text segments (for BM25 / dense retrieval)

### 2) Visual Page View
- Rendered page images
- Page-level multivector embeddings (ColQwen2)

This allows covering both **pure text search** and **visual/graphical retrieval** without forcing all queries through expensive visual pipelines.

---

## 🚀 Quick Start (Local Development)

### 1) Create and activate Python environment
```bash
uv python install 3.10
uv venv --python 3.10 .venv-colbert
source .venv-colbert/bin/activate

source colbert-env/.venv/bin/activate
uv sync --project colbert-env
deactivate

source colpali-env/.venv/bin/activate
uv sync --project colpali-env
deactivate

python -m uvicorn src.main_colbert:app --reload
```

### 2) Install dependencies
```bash
uv sync --project colbert-env
uv sync --project colpali-env
mkdir -p external
git clone https://github.com/sionic-ai/muvera-py.git external/muvera-py
```

Do not install `colbert-ai` and `colpali-engine` into the same environment. `colpali-engine` requires `transformers>=5`, while the official ColBERT path in this repo needs the dedicated `colbert-env` project.

### 3) Run the API server
```bash
uv run --project colbert-env uvicorn src.main_colbert:app --reload
uv run --project colpali-env uvicorn src.main_colpali:app --reload

source .venv-colbert/bin/activate
python -m uvicorn src.main_colbert:app --reload

source .venv-colpali/bin/activate
python -m uvicorn src.main_colpali:app --reload
```

`src.main_colbert:app` includes the text endpoints and the official ColBERT experimental routes. `src.main_colpali:app` includes the text endpoints plus the visual endpoints, and only loads the visual model when a visual endpoint is called.

`src.main:app` is kept as a text-only compatibility alias to `src.main_colbert:app`.

For ColBERT endpoints, do not use `uv run uvicorn src.main:app --reload` from the repository root. That command resolves the root project dependency set and can pull in `transformers>=5`, which breaks `colbert-ai`.

---

## 🧪 Basic API Usage

### Upload / Ingest a PDF
```bash
curl -X POST "http://127.0.0.1:8000/ingest/pdf" \
  -F "file=@$/home/ubuntu/file-sample_150kB.pdf"
```

### Search (text retrieval)
```bash
curl --get "http://127.0.0.1:8000/search" \
  --data-urlencode "q=prompt" \
  --data-urlencode "top_k=5" | python -m json.tool
```

### Answer With Citations
```bash
curl --get "http://127.0.0.1:8000/answer" \
  --data-urlencode "q=what is the termination notice period?" \
  --data-urlencode "top_k=5" \
  --data-urlencode "evidence_k=3" | python -m json.tool
```

### Visual Search (page images + ColQwen2)
```bash
curl -X POST "http://127.0.0.1:8000/visual/embed-pages" | python -m json.tool

curl --get "http://127.0.0.1:8000/visual/search" \
  --data-urlencode "q=prompt" \
  --data-urlencode "top_k=5" | python -m json.tool
```

### Debug: Inspect Page Records
```bash
curl "http://127.0.0.1:8000/debug/pages" | python -m json.tool


```


Step 10: what to run

After you already uploaded and indexed documents into LanceDB, run:

curl -X POST "http://127.0.0.1:8000/experimental/colbert/reindex" | python3 -m json.tool

Or start it in the background and poll progress:

curl -X POST "http://127.0.0.1:8000/experimental/colbert/reindex/background" | python3 -m json.tool

curl "http://127.0.0.1:8000/experimental/colbert/reindex/status" | python3 -m json.tool

Then compare:

curl --get "http://127.0.0.1:8000/experimental/search" \
  --data-urlencode "q=signatures" \
  --data-urlencode "top_k=5" | python -m json.tool

That gives you:

BM25

dense

hybrid

real ColBERT retrieval

using the official ColBERT index/search APIs.
---

## 🧭 Milestones (Roadmap)

### Milestone 1 – Core Ingestion + API
- ✅ FastAPI running with `/health`
- ✅ PDF extraction → text + page images
- ✅ Basic test coverage

### Milestone 2 – Baseline Retrieval
- BM25 retrieval (text)
- Dense retrieval (text)
- Hybrid RRF fusion
- Single `/search` endpoint

### Milestone 3 – Visual Page Store
- Store page metadata (page number, PDF reference, etc.)
- Store page image paths
- Store placeholder visual embeddings
- One `/ingest/pdf` endpoint

### Milestone 4 – ColQwen2 Embeddings (Visual Retrieval)
- Query embeddings (ColQwen2)
- Page-image embeddings (ColQwen2)
- Page-level candidate scoring

> **Design note:** For now, we do **not** store full multi-vectors in LanceDB. Instead:
> - Pool ColQwen2 page embeddings into a single page vector
> - Pool ColQwen2 query embeddings into one vector
> - Use page-level vector search for first-stage retrieval
>
> This enables real ColQwen2 retrieval without full late interaction—while leaving a clean path to full multi-vector scoring later.

### Milestone 5 – ColBERT Reranking (Text)
- Rerank top chunks using ColBERT (late interaction)
- Compare baseline vs. reranked results

> Recommended approach:
> - Start with reranking only (no full ColBERT index)
> - Rerank top 20–50 chunks (fast and easy)
> - Provides immediate quality improvement with minimal engineering overhead

---

## 📌 Notes & Tips
- The system uses a **hybrid retrieval stack** (text + visual) to avoid forcing every query through the expensive visual pipeline.
- The routing logic is handled by **LangGraph** to decide which retrieval paths to run.

---

## ✅ Next Steps
- Make sure `uvicorn` is running
- Upload a PDF with `/ingest/pdf`
- Run `/search` and `/visual/search`
- Inspect `/debug/pages` to verify ingestion

---

## 📄 References
- ColQwen2 via `colpali-engine` (multi-vector embeddings)
- ColBERT reranking (late interaction)
- LanceDB (vector store + retrieval)


Add MUVERA proxy stage:

plug FDE/proxy encoding behind MuveraProxyIndex

use it only for candidate generation

keep ColBERT/ColQwen2 for final rerank

Important design decision

Do not try to store raw ColBERT token embeddings inside LanceDB first.

For your first real test:

store muvera_vector in LanceDB

keep colbert_vectors on disk as files keyed by chunk id

Why:

LanceDB is good for single-vector retrieval

ColBERT multivectors are variable-length token embeddings

storing them as sidecar .npy or .pt files is simpler for experimentation

So a good storage layout is:

data/
  lancedb/
  colbert_vectors/
    <chunk_id>.pt

Then:

MUVERA retrieves candidate chunk IDs from LanceDB

ColBERT reranker loads the .pt token embeddings for those candidate IDs

scores them against the query token embeddings

That gives you the real algorithmic behavior without fighting the database.

Real Milestone 6 flow
Ingestion

For each chunk:

compute dense vector

compute ColBERT token embeddings

save token embeddings to disk

compute MUVERA FDE from those token embeddings

store row in LanceDB with:

id

chunk_text

vector

muvera_vector

metadata

Search

BM25 hits

dense hits

MUVERA hits from muvera_vector

union candidates

ColBERT rerank candidates using saved token embeddings

return final ranked list


Milestone 7

Add answer generation and citations:

retrieve top evidence

build prompt

return cited answer

That sequence keeps the project usable at every step.


==============
I would make your first week look like this:

Day 1

initialize with uv

create folders

make FastAPI run

write chunking and PDF extraction

Day 2

build BM25

build dense retriever

add hybrid fusion

test on 2–3 sample PDFs

Day 3

add page-image indexing schema

store image paths + page metadata

expose /ingest/pdf

Day 4

add ColQwen2 wrapper interface

add ColBERT wrapper interface

implement router

Day 5

wire LangGraph around the flow

add logs and debug payloads

start evaluation dataset

That gets you a working skeleton without hiding the architecture.



oncrete target architecture
User query
  -> SearchService
      -> BM25
      -> dense
      -> hybrid RRF
      -> MUVERA proxy candidates (later)
      -> ColBERTReranker
      -> ColQwen2 visual reranker (for page queries)
  -> AnswerService
      -> prompt builder
      -> grounded answer
      -> citations

  

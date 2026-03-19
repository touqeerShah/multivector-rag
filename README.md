# Multivector RAG

Multivector RAG is a FastAPI project for PDF retrieval and grounded answering. It supports:

- text ingestion and retrieval
- visual page retrieval with ColQwen2
- official ColBERT indexing and search
- experimental MUVERA candidate retrieval
- cited answer generation from retrieved evidence

## Overview

The project is split into two runtime apps because the dependency stacks are different:

- `src.main_colpali:app`
  Use for text endpoints plus visual page embedding and visual search.
- `src.main_colbert:app`
  Use for text endpoints plus official ColBERT and real ColBERT-backed MUVERA experiments.

Both apps share the same data directories, so you can ingest documents in one app and search them from the other.

## Project Layout

For a file-by-file explanation of the codebase and request flow, see:

- [docs/project-structure.md](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/docs/project-structure.md)
- [docs/real-colbert-muvera.md](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/docs/real-colbert-muvera.md)

## Prerequisites

- Python `3.11`
- [`uv`](https://docs.astral.sh/uv/)
- `git`

Optional:

- Tesseract language data if you want OCR support during PDF extraction

## Setup

### 1. Install dependencies

```bash
uv sync --project colbert-env
uv sync --project colpali-env
```

### 2. Install MUVERA dependency

```bash
mkdir -p external
git clone https://github.com/sionic-ai/muvera-py.git external/muvera-py
```

### 3. Important environment rule

Do not install `colbert-ai` and `colpali-engine` into the same Python environment.

- `colpali-engine` requires newer `transformers`
- the official ColBERT path in this repo requires the dedicated `colbert-env`

## Run The Apps

### ColPali App

Use this for:

- `/upload`
- `/search`
- `/answer`
- `/ingest/pdf`
- `/debug/pages`
- `/visual/embed-pages`
- `/visual/search`
- `/experimental/muvera/reindex`
- `/experimental/muvera/search`

Run:

```bash
source colpali-env/.venv/bin/activate
python -m uvicorn src.main_colpali:app --reload --port 8000
```

Or with `uv`:

```bash
uv run --project colpali-env uvicorn src.main_colpali:app --reload --port 8000
```

### ColBERT App

Use this for:

- `/upload`
- `/search`
- `/answer`
- `/experimental/colbert/reindex`
- `/experimental/colbert/reindex/background`
- `/experimental/colbert/reindex/status`
- `/experimental/search`
- `/experimental/muvera/real/reindex`
- `/experimental/muvera/real/search`

Run:

```bash
source colbert-env/.venv/bin/activate
python -m uvicorn src.main_colbert:app --reload --port 8001
```

Or with `uv`:

```bash
uv run --project colbert-env uvicorn src.main_colbert:app --reload --port 8001
```

Do not run:

```bash
uv run uvicorn src.main:app --reload
```

from the repository root for ColBERT work. That can resolve the wrong dependency set and break the official ColBERT runtime.

## Endpoint Summary

### Shared text endpoints

Available on both apps:

- `GET /health`
- `POST /upload`
- `GET /search`
- `GET /answer`
- `GET /debug/rows`
- `POST /experimental/muvera/reindex`
- `GET /experimental/muvera/search`

### Visual endpoints

Available on `src.main_colpali:app`:

- `POST /ingest/pdf`
- `GET /debug/pages`
- `POST /visual/embed-pages`
- `GET /visual/search`

### ColBERT-only endpoints

Available on `src.main_colbert:app`:

- `POST /experimental/colbert/reindex`
- `POST /experimental/colbert/reindex/background`
- `GET /experimental/colbert/reindex/status`
- `GET /experimental/search`
- `POST /experimental/muvera/real/reindex`
- `GET /experimental/muvera/real/search`

## Recommended API Flow

### 1. Text flow

Run against the ColPali app on port `8000` or the ColBERT app on port `8001`.

Health check:

```bash
curl "http://127.0.0.1:8000/health" | python -m json.tool
```

Upload and index a file:

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@/absolute/path/to/document.pdf" | python -m json.tool
```

Inspect stored text rows:

```bash
curl "http://127.0.0.1:8000/debug/rows?limit=5" | python -m json.tool
```

Text search:

```bash
curl --get "http://127.0.0.1:8000/search" \
  --data-urlencode "q=your question" \
  --data-urlencode "top_k=5" | python -m json.tool
```

Answer with citations:

```bash
curl --get "http://127.0.0.1:8000/answer" \
  --data-urlencode "q=your question" \
  --data-urlencode "top_k=5" \
  --data-urlencode "evidence_k=3" | python -m json.tool
```

### 2. Visual flow

Run against the ColPali app on port `8000`.

Ingest page records and images:

```bash
curl -X POST "http://127.0.0.1:8000/ingest/pdf" \
  -F "file=@/absolute/path/to/document.pdf" | python -m json.tool
```

Inspect page rows:

```bash
curl "http://127.0.0.1:8000/debug/pages?limit=5" | python -m json.tool
```

Embed stored page images:

```bash
curl -X POST "http://127.0.0.1:8000/visual/embed-pages" | python -m json.tool
```

Visual search:

```bash
curl --get "http://127.0.0.1:8000/visual/search" \
  --data-urlencode "q=your visual question" \
  --data-urlencode "top_k=5" | python -m json.tool
```

Notes:

- page embeddings are stored in the page table as `visual_vector`
- API responses expose page metadata and scores, not raw vectors
- `GET /debug/pages` is the easiest way to verify that page embeddings were created

### 3. Proxy MUVERA flow

Run against the ColPali app on port `8000` or the ColBERT app on port `8001`.

Build the proxy MUVERA index:

```bash
curl -X POST "http://127.0.0.1:8000/experimental/muvera/reindex" \
  --get --data-urlencode "max_subvectors_per_doc=8" | python -m json.tool
```

Search with proxy MUVERA:

```bash
curl --get "http://127.0.0.1:8000/experimental/muvera/search" \
  --data-urlencode "q=your question" \
  --data-urlencode "top_k=5" \
  --data-urlencode "max_query_subvectors=6" | python -m json.tool
```

This endpoint compares:

- `muvera`
- `dense`
- `hybrid`

### 4. Official ColBERT flow

Run against the ColBERT app on port `8001`.

Build the official ColBERT index:

```bash
curl -X POST "http://127.0.0.1:8001/experimental/colbert/reindex" | python -m json.tool
```

Or run it in the background:

```bash
curl -X POST "http://127.0.0.1:8001/experimental/colbert/reindex/background" | python -m json.tool
```

Poll background status:

```bash
curl "http://127.0.0.1:8001/experimental/colbert/reindex/status" | python -m json.tool
```

Search with the official ColBERT index:

```bash
curl --get "http://127.0.0.1:8001/experimental/search" \
  --data-urlencode "q=your question" \
  --data-urlencode "top_k=5" | python -m json.tool
```

### 5. Real ColBERT-backed MUVERA flow

Run against the ColBERT app on port `8001`.

Build the real MUVERA index from saved ColBERT document multivectors:

```bash
curl -X POST "http://127.0.0.1:8001/experimental/muvera/real/reindex" | python -m json.tool
```

Optional smaller build:

```bash
curl -X POST "http://127.0.0.1:8001/experimental/muvera/real/reindex" \
  --get \
  --data-urlencode "top_docs=50" \
  --data-urlencode "batch_size=8" | python -m json.tool
```

Search with real ColBERT-backed MUVERA:

```bash
curl --get "http://127.0.0.1:8001/experimental/muvera/real/search" \
  --data-urlencode "q=your question" \
  --data-urlencode "top_k=5" \
  --data-urlencode "rerank_k=10" | python -m json.tool
```

This endpoint returns:

- `muvera_candidates`
- `reranked`
- `proxy_muvera`
- `dense`
- `hybrid`

It also includes diagnostic fields inside reranked hits such as:

- `best_query_variant`
- `best_variant_maxsim_score`
- `query_term_coverage`
- `answerability_score`
- `reference_penalty`
- `composite_score`

## Example End-to-End Sequence

### Terminal 1: ColPali app

```bash
source colpali-env/.venv/bin/activate
python -m uvicorn src.main_colpali:app --reload --port 8000
```

### Terminal 2: ColBERT app

```bash
source colbert-env/.venv/bin/activate
python -m uvicorn src.main_colbert:app --reload --port 8001
```

### Example request order

1. `POST /upload`
2. `GET /debug/rows`
3. `GET /search`
4. `GET /answer`
5. `POST /ingest/pdf`
6. `GET /debug/pages`
7. `POST /visual/embed-pages`
8. `GET /visual/search`
9. `POST /experimental/muvera/reindex`
10. `GET /experimental/muvera/search`
11. `POST /experimental/colbert/reindex`
12. `GET /experimental/search`
13. `POST /experimental/muvera/real/reindex`
14. `GET /experimental/muvera/real/search`

## Storage

Important generated data:

- `data/raw`
  Uploaded source files
- `data/processed`
  Extracted page assets
- `data/lancedb`
  Text rows and page rows
- `data/colbert`
  ColBERT collection export files
- `data/muvera`
  Proxy MUVERA vectors
- `data/muvera_real`
  Real ColBERT-backed MUVERA vectors
- `data/colbert_vectors`
  Saved ColBERT document multivectors as `.pt` files

## Notes

- `/upload` supports `.pdf`, `.txt`, and `.md`
- `/ingest/pdf` supports `.pdf` only
- heavy models are loaded lazily where possible
- raw text and visual vectors are intentionally omitted from the main API-facing search responses

## Development Notes

Useful tests:

```bash
source colbert-env/.venv/bin/activate
pytest tests/test_search_service.py tests/test_experimental_real_muvera_service.py tests/test_app_entrypoints.py
```

If you are onboarding to the codebase, start with:

- [docs/project-structure.md](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/docs/project-structure.md)
- [src/api/text_routes.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/src/api/text_routes.py)
- [src/api/visual_routes.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/src/api/visual_routes.py)

# Real ColBERT-Backed MUVERA

This document explains what was added, how it works, and how it differs from the earlier proxy MUVERA path.

## Goal

The project already had an experimental MUVERA pipeline:

- split each chunk into mini-spans
- embed spans with the dense text embedder
- treat those span embeddings as a multivector
- compress them with MUVERA fixed-dimensional encoding
- search over the compressed vectors

That path is useful for learning and behavior testing, but it is not the final ColBERT-style multivector pipeline.

The new work adds a second experimental path that uses real ColBERT document/query multivectors before MUVERA compression.

## What Was Added

### New service

File: [src/services/experimental_real_muvera_service.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/src/services/experimental_real_muvera_service.py)

This service does four things:

1. loads text rows from the retrieval store
2. encodes each row with official ColBERT document embeddings
3. saves the raw ColBERT multivectors to disk
4. builds MUVERA fixed-dimensional vectors from those saved multivectors

It also supports search:

1. encode query with ColBERT query embeddings
2. compress query multivector with MUVERA
3. retrieve candidates from the MUVERA index
4. rerank those candidates with real ColBERT MaxSim
5. compare results against proxy MUVERA and the normal dense/hybrid search path

### New API endpoints

File: [src/api/text_routes.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/src/api/text_routes.py)

Added under the ColBERT app only:

- `POST /experimental/muvera/real/reindex`
- `GET /experimental/muvera/real/search`

These routes are available in:

- [src/main_colbert.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/src/main_colbert.py)

They are not available in:

- [src/main_colpali.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/src/main_colpali.py)

### Tests

Files:

- [tests/test_experimental_real_muvera_service.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/tests/test_experimental_real_muvera_service.py)
- [tests/test_app_entrypoints.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/tests/test_app_entrypoints.py)

The tests verify:

- ColBERT multivectors are saved to disk during reindex
- MUVERA vectors are built from those multivectors
- search returns reranked ColBERT-backed results
- raw vectors are not leaked in API response payloads
- the new routes exist only on the ColBERT app

## File Connections

### Input data

[src/retrieval/store.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/src/retrieval/store.py)

The retrieval store is still the source of truth for text rows. The real MUVERA service reads chunk records from here.

### ColBERT model access

[src/retrieval/colbert_service.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/src/retrieval/colbert_service.py)

This file already contained ColBERT runtime compatibility checks. The new real MUVERA service reuses that compatibility validation before loading ColBERT checkpoints.

### MUVERA encoding

[src/retrieval/muvera_encoder.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/src/retrieval/muvera_encoder.py)

This file is now used in a live experimental path. It converts variable-length ColBERT multivectors into a fixed-dimensional encoding.

### MUVERA storage

[src/retrieval/muvera_store.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/src/retrieval/muvera_store.py)

This file is also now used in a live experimental path. It stores the compressed MUVERA vectors and performs nearest-neighbor retrieval on them.

### Proxy comparison

[src/services/experimental_muvera_service.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/src/services/experimental_muvera_service.py)

The new real MUVERA service compares its output against this earlier proxy path so you can inspect ranking differences.

### Dense and hybrid comparison

[src/services/search_service.py](/Users/touqeershah/Documents/PharmaTraceProject-Files/multivector-rag/src/services/search_service.py)

The real MUVERA endpoint also compares against the normal dense/hybrid text search output from this service.

## How Reindex Works

`POST /experimental/muvera/real/reindex`

Sequence:

1. read text chunks from the main retrieval store
2. optionally limit to `top_docs`
3. load the official ColBERT checkpoint
4. call ColBERT document encoding for each chunk
5. save each document multivector to `data/colbert_vectors/*.pt`
6. pass each multivector into `MuveraEncoder`
7. save the resulting fixed vectors with `MuveraStore`

Outputs on disk:

- raw ColBERT multivectors: `data/colbert_vectors/*.pt`
- MUVERA vectors: `data/muvera_real/doc_vectors.npy`
- MUVERA ids: `data/muvera_real/doc_ids.json`

The exact filename inside `data/colbert_vectors` is based on document id plus a hash, so file paths stay stable and safe.

## How Search Works

`GET /experimental/muvera/real/search?q=...`

Sequence:

1. encode query with ColBERT query embeddings
2. compress query multivector with MUVERA
3. retrieve top candidates from the MUVERA index
4. load saved ColBERT document multivectors for those candidates
5. rerank them using ColBERT MaxSim
6. fetch comparison results from:
   - proxy MUVERA
   - dense search
   - hybrid search

Response sections:

- `muvera_candidates`
- `reranked`
- `proxy_muvera`
- `dense`
- `hybrid`
- `overlap`
- `config`
- `notes`

The API does not return raw vector payloads. It returns metadata, scores, and references to stored ColBERT vector files where useful.

## Why This Is Better Than Proxy MUVERA

### Proxy MUVERA

Uses mini-span dense embeddings from the regular text embedder.

Good for:

- fast experimentation
- understanding the MUVERA compression step
- rough ranking comparison

Weakness:

- not aligned with ColBERT token-level late interaction

### Real ColBERT-Backed MUVERA

Uses official ColBERT document/query multivectors.

Better for:

- testing a realistic ColBERT candidate-retrieval stage
- storing compressed candidates while preserving ColBERT semantics better
- reranking with real ColBERT MaxSim after MUVERA candidate retrieval

Remaining limitation:

- this is still an experimental path, not a fully productionized end-to-end index format

## API Usage

Start the ColBERT app:

```bash
source .venv-colbert/bin/activate
python -m uvicorn src.main_colbert:app --reload --port 8000
```

Build the real MUVERA index:

```bash
curl -X POST "http://127.0.0.1:8000/experimental/muvera/real/reindex" | python3 -m json.tool
```

Optional smaller build:

```bash
curl -X POST "http://127.0.0.1:8000/experimental/muvera/real/reindex?top_docs=50&batch_size=8" | python -m json.tool
```

Search:

```bash
curl --get "http://127.0.0.1:8001/experimental/muvera/real/search" \
  --data-urlencode "q=your question" \
  --data-urlencode "top_k=5" \
  --data-urlencode "rerank_k=10" | python -m json.tool
```

## Practical Comparison

If you want to compare the three paths for one query, call:

1. `GET /search`
2. `GET /experimental/muvera/search`
3. `GET /experimental/muvera/real/search`

Interpret them as:

- `/search`: current dense + hybrid text retrieval
- `/experimental/muvera/search`: MUVERA over proxy dense mini-span multivectors
- `/experimental/muvera/real/search`: MUVERA over real ColBERT multivectors, then ColBERT rerank

## Verification

The targeted test suite passed after this change:

- `tests/test_experimental_real_muvera_service.py`
- `tests/test_experimental_muvera_service.py`
- `tests/test_app_entrypoints.py`
- `tests/test_search_service.py`
- `tests/test_answer_service.py`
- `tests/test_indexing_service.py`
- `tests/test_experimental_text_indexing.py`
- `tests/test_colbert_service.py`
- `tests/test_chunking.py`
- `tests/test_router.py`

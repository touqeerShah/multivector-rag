# Multivector RAG Project Structure

## Purpose

This document explains:

- what each important file does
- how files connect to each other
- what runs in `ColBERT` mode vs `ColPali` mode
- where settings come from
- how data moves through the project

The project is split into two runtime apps because the dependency stacks are different:

- `ColBERT` app: text retrieval + official ColBERT experimental indexing/search
- `ColPali` app: text retrieval + visual page embedding/search

Both apps share the same project data directories, especially:

- `data/raw`
- `data/processed`
- `data/lancedb`
- `data/colbert`
- `data/muvera`
- `data/muvera_real`
- `data/colbert_vectors`

That means you can ingest in one app and use the stored data in the other app.

---

## Runtime Entry Points

### `src/main_colbert.py`

- Creates the FastAPI app for the ColBERT environment.
- Includes only the text router.
- Enables the official experimental ColBERT endpoints.

Flow:

- `FastAPI`
- `build_text_router(include_official_colbert=True)`

Use this when running:

```bash
python -m uvicorn src.main_colbert:app --reload
```

### `src/main_colpali.py`

- Creates the FastAPI app for the ColPali environment.
- Includes the text router.
- Includes the visual router.
- Does not expose the official ColBERT experimental routes.

Flow:

- `FastAPI`
- `build_text_router(include_official_colbert=False)`
- `visual_router`

Use this when running:

```bash
python -m uvicorn src.main_colpali:app --reload
```

### `src/main.py`

- Compatibility alias.
- Imports `app` from `src.main_colbert`.
- Useful for older commands, but the explicit entrypoints above are preferred.

### `src/api/routes.py`

- Compatibility shim.
- Builds the text router with ColBERT experimental routes enabled.
- Kept so older imports do not break.

---

## Environment Configuration

### `colbert-env/pyproject.toml`

- Defines the ColBERT-specific Python environment.
- Includes:
  - `colbert-ai`
  - `faiss-cpu`
  - `sentence-transformers>=3,<4`
  - `transformers==4.49.0`

Why:

- official ColBERT in this project needs the older Transformers stack
- this environment is for text retrieval and official ColBERT indexing/search

### `colpali-env/pyproject.toml`

- Defines the ColPali-specific Python environment.
- Includes:
  - `colpali-engine`
  - `sentence-transformers>=5.3.0`
  - `transformers>=5,<6`

Why:

- ColPali / ColQwen2 depends on a newer Transformers stack
- this environment is for visual embedding and page-image search

### `src/core/config.py`

- Central runtime settings object.
- Controls:
  - app name
  - raw upload directory
  - processed file directory
  - LanceDB directory
  - text table name

Computed properties:

- `raw_path`
- `processed_path`
- `lancedb_path`

These properties create directories automatically.

---

## API Layer

### `src/api/text_routes.py`

- Defines the text API.
- Lazily creates heavy services so app import is cheap.

Endpoints:

- `GET /health`
- `POST /upload`
- `GET /search`
- `GET /answer`
- `GET /debug/rows`
- `POST /experimental/muvera/reindex`
- `GET /experimental/muvera/search`

ColBERT-only endpoints:

- `POST /experimental/colbert/reindex`
- `POST /experimental/colbert/reindex/background`
- `GET /experimental/colbert/reindex/status`
- `GET /experimental/search`
- `POST /experimental/muvera/real/reindex`
- `GET /experimental/muvera/real/search`

Main dependencies:

- `DenseEmbedder`
- `IndexingService`
- `SearchService`
- `AnswerService`
- `ExperimentalMuveraService`
- `ExperimentalTextIndexingService`
- `ExperimentalSearchService`
- `ExperimentalRealMuveraService`

Connection summary:

- upload -> indexing service -> retrieval store
- search -> search service -> bm25 + dense + rerank
- answer -> answer service -> search service
- proxy MUVERA reindex -> dense mini-span multivectors -> MUVERA fixed vectors
- experimental reindex -> export collection -> official ColBERT index build
- real MUVERA reindex -> ColBERT multivectors -> MUVERA fixed vectors -> optional ColBERT rerank

### `src/api/visual_routes.py`

- Defines the visual API.

Endpoints:

- `POST /ingest/pdf`
- `GET /debug/pages`
- `POST /visual/embed-pages`
- `GET /visual/search`

Main dependencies:

- `PageIndexingService`
- `VisualIndexingService`

Connection summary:

- ingest pdf -> extract page markdown/images -> store page rows
- embed pages -> ColQwen2 pooled page vectors -> update page table
- visual search -> ColQwen2 query vector -> page vector search -> rerank by dot product

---

## Service Layer

### `src/services/indexing.py`

- Handles text indexing for `/upload`.
- Supports `.pdf`, `.txt`, and `.md`.

For PDF:

- extract markdown and images per page
- create semantic chunks
- embed chunks with the dense embedder
- store rows in LanceDB text table

For text files:

- read file text
- split into chunks
- embed chunks
- store rows in LanceDB text table

Also rebuilds BM25 after insert.

Uses:

- `src/ingest/pdf.py`
- `src/ingest/chunking.py`
- `src/retrieval/store.py`
- `src/retrieval/bm25.py`

### `src/services/search_service.py`

- Main text retrieval service.
- Used by `/search` and indirectly by `/answer`.

Pipeline:

1. load all text rows from LanceDB
2. build BM25
3. dense vector search
4. hybrid fusion with RRF
5. rerank top candidates

Returns:

- `bm25`
- `dense`
- `hybrid`
- `reranked`

### `src/services/answer_service.py`

- Milestone 7 answer generation service.
- Builds a usable answer without requiring an external LLM.

Pipeline:

1. call `SearchService.search()`
2. choose top evidence rows
3. build a grounded prompt string
4. extract the best matching sentences
5. return answer + citations + prompt + evidence + retrieval payload

This is currently extractive and deterministic.

### `src/services/rerank_service.py`

- Thin wrapper around the current ColBERT-style reranker.
- Exists so reranking can be replaced later without changing the search service.

### `src/services/page_indexing.py`

- Stores per-page page-image rows for PDFs.
- Creates placeholder visual vectors first.
- Used by `/ingest/pdf`.

### `src/services/visual_indexing.py`

- Handles visual embedding and visual search.
- Lazily imports `ColQwen2Service`, so the ColBERT app can still boot without ColPali dependencies.

Responsibilities:

- embed stored page images
- update `visual_vector` and `visual_status`
- search the page table with a visual query vector
- return ranked visual page hits

### `src/services/experimental_text_indexing.py`

- Builds the official ColBERT index from the stored text rows.

Pipeline:

1. load text rows from LanceDB
2. export `data/colbert/collection.tsv`
3. start official ColBERT indexing
4. track status/logs for foreground or background runs

Used by:

- `/experimental/colbert/reindex`
- `/experimental/colbert/reindex/background`
- `/experimental/colbert/reindex/status`

### `src/services/experimental_search_service.py`

- Uses the official ColBERT index for search.
- Still compares against BM25, dense, and hybrid results.

Used by:

- `/experimental/search`

### `src/services/experimental_muvera_service.py`

- Experimental proxy MUVERA path.
- Builds MUVERA vectors from dense mini-span embeddings rather than official ColBERT token embeddings.

Pipeline:

1. load text rows from LanceDB
2. split each chunk into mini-spans
3. embed those spans with the dense text embedder
4. compress each document multivector with MUVERA
5. save/search the compressed vectors with `MuveraStore`

Used by:

- `/experimental/muvera/reindex`
- `/experimental/muvera/search`

### `src/services/experimental_real_muvera_service.py`

- Experimental real ColBERT-backed MUVERA path.
- Uses official ColBERT document/query multivectors before MUVERA compression.

Pipeline:

1. load text rows from LanceDB
2. encode each chunk with the ColBERT checkpoint
3. save raw ColBERT multivectors to disk
4. compress each multivector with MUVERA
5. retrieve candidates from MUVERA
6. rerank those candidates with real ColBERT MaxSim
7. compare against proxy MUVERA and normal dense/hybrid search

Used by:

- `/experimental/muvera/real/reindex`
- `/experimental/muvera/real/search`

---

## Retrieval Layer

### `src/retrieval/store.py`

- Shared persistence layer built on LanceDB.
- Stores both:
  - text chunk rows
  - page image rows

Text table fields:

- chunk id
- doc id
- source file
- page and section metadata
- chunk text
- dense vector

Page table fields:

- page id
- doc id
- source file
- page number
- image path
- markdown
- preview text
- visual vector
- visual status

Main methods:

- `add_text_rows()`
- `add_page_rows()`
- `all_text_rows()`
- `all_page_rows()`
- `text_vector_search()`
- `page_vector_search()`

### `src/retrieval/dense.py`

- Wraps `SentenceTransformer`.
- Provides:
  - `embed_texts()`
  - `embed_query()`

Used by:

- `IndexingService`
- `SearchService`
- `ExperimentalSearchService`

### `src/retrieval/bm25.py`

- Simple BM25 index over `chunk_text`.
- Rebuilt in memory from stored rows when needed.

### `src/retrieval/hybrid.py`

- Implements reciprocal rank fusion.
- Merges BM25 and dense result lists into one combined ranking.

### `src/retrieval/collection_export.py`

- Converts stored text rows into ColBERT’s TSV input format.
- Also writes PID to internal chunk ID mapping.

Outputs:

- `data/colbert/collection.tsv`
- `data/colbert/pid_mapping.json`

### `src/retrieval/colbert_service.py`

- Official ColBERT integration.
- Only valid in the ColBERT environment.

Responsibilities:

- environment compatibility checks
- official ColBERT index building
- dynamic partition selection for small/medium/large corpora
- background-friendly logging hooks
- official ColBERT search

Important behavior:

- partition count is chosen dynamically using exported row count and estimated embeddings
- final partition count is clamped so FAISS training does not fail on tiny corpora

### `src/retrieval/muvera_encoder.py`

- Experimental MUVERA fixed-dimensional encoding wrapper.
- Imports local code from `external/muvera-py`.
- Now used by both experimental MUVERA HTTP paths:
  - proxy MUVERA
  - real ColBERT-backed MUVERA

### `src/retrieval/muvera_store.py`

- Experimental storage/search layer for MUVERA vectors.
- Saves vectors to `.npy` and IDs to `.json`.
- Now used by both experimental MUVERA HTTP paths.
- Stores:
  - `data/muvera/*` for proxy MUVERA
  - `data/muvera_real/*` for real ColBERT-backed MUVERA

---

## Ingestion Layer

### `src/ingest/pdf.py`

- Extracts page markdown and plain text from PDFs.
- Renders each PDF page to an image file.

Outputs per page:

- page number
- markdown
- plain text
- image path
- source file

Used by:

- `IndexingService`
- `PageIndexingService`

### `src/ingest/chunking.py`

- Shared text chunking utilities.
- Supports:
  - plain text chunking
  - markdown heading splitting
  - section-aware chunk creation

Used by:

- `IndexingService`

### `src/ingest/images.py`

- Currently empty placeholder.
- Reserved for future image-specific ingestion utilities.

### `src/ingest/text.py`

- Currently empty placeholder.
- Reserved for future text-specific ingestion helpers.

---

## Reranking Layer

### `src/rerank/colbert_reranker.py`

- Current text reranker used in the main `/search` path.
- This is not the official ColBERT engine.
- It is a lightweight lexical/heading-aware scoring approximation used as a practical reranking stage.

### `src/rerank/colqwen.py`

- Visual encoder/reranker wrapper around `colpali_engine`.
- Loads `ColQwen2` and `ColQwen2Processor`.
- Produces pooled visual vectors for:
  - queries
  - page images

Used by:

- `VisualIndexingService`

### `src/rerank/router.py`

- Tiny query classifier.
- Decides if a query looks visual or text-oriented.
- Mostly used by the experimental graph/stub workflow.

### `src/rerank/colbert.py`

- Old interface stub for a future direct ColBERT reranker.
- Not part of the main active runtime path.

### `src/rerank/rerank`

- Empty placeholder file.
- Safe to ignore right now.

---

## Graph Layer

### `src/graph/state.py`

- Defines the `GraphState` shape for LangGraph experiments.

### `src/graph/workflow.py`

- Experimental LangGraph pipeline stub.
- Stages:
  - classify
  - retrieve
  - rerank
  - answer

Current status:

- not the main serving path
- useful as a future orchestration shell

---

## Core / Package Files

### `src/__init__.py`

- Package marker for the `src` package.

### `src/app/__init__.py`

- Currently empty.
- Reserved for future app-level packaging if the project grows.

### `src/core/types.py`

- Currently empty.
- Reserved for shared typed models if needed later.

---

## Tests

### `tests/test_answer_service.py`

- Verifies prompt, answer, and citation output from `AnswerService`.

### `tests/test_app_entrypoints.py`

- Verifies route exposure for:
  - `main_colbert`
  - `main_colpali`

### `tests/test_chunking.py`

- Verifies chunk splitting.

### `tests/test_colbert_service.py`

- Verifies ColBERT environment checks and dynamic partition logic.

### `tests/test_experimental_text_indexing.py`

- Verifies ColBERT reindex status/log tracking.

### `tests/test_experimental_muvera_service.py`

- Verifies the proxy MUVERA reindex/search path.

### `tests/test_experimental_real_muvera_service.py`

- Verifies real ColBERT multivectors are saved and used for MUVERA retrieval + ColBERT rerank.

### `tests/test_indexing_service.py`

- Verifies text indexing uses the correct store API.

### `tests/test_router.py`

- Verifies simple text vs visual query classification.

---

## Request Flow Summary

## When running `src.main_colbert:app`

Available paths:

- `/health`
- `/upload`
- `/search`
- `/answer`
- `/debug/rows`
- `/experimental/muvera/reindex`
- `/experimental/muvera/search`
- `/experimental/colbert/reindex`
- `/experimental/colbert/reindex/background`
- `/experimental/colbert/reindex/status`
- `/experimental/search`
- `/experimental/muvera/real/reindex`
- `/experimental/muvera/real/search`

What happens on `/upload`:

1. `text_routes.py`
2. `IndexingService`
3. `extract_pdf_markdown_and_images()` or `extract_txt_text()`
4. `chunking.py`
5. `DenseEmbedder`
6. `RetrievalStore.add_text_rows()`
7. `BM25Index.build()`

What happens on `/search`:

1. `text_routes.py`
2. `SearchService`
3. BM25 + dense vector search
4. `hybrid.py`
5. `RerankService`
6. return ranked candidates

What happens on `/answer`:

1. `text_routes.py`
2. `AnswerService`
3. `SearchService`
4. evidence selection
5. prompt building
6. cited answer response

What happens on `/experimental/colbert/reindex`:

1. `text_routes.py`
2. `ExperimentalTextIndexingService`
3. `CollectionExporter`
4. `OfficialColBERTService.build_index()`
5. ColBERT writes index under `experiments/local/indexes/local_index`

What happens on `/experimental/muvera/reindex`:

1. `text_routes.py`
2. `ExperimentalMuveraService`
3. load stored text rows
4. split chunks into mini-spans
5. embed spans with `DenseEmbedder`
6. `MuveraEncoder`
7. `MuveraStore`
8. return proxy MUVERA index metadata

What happens on `/experimental/muvera/real/reindex`:

1. `text_routes.py`
2. `ExperimentalRealMuveraService`
3. load stored text rows
4. encode docs with official ColBERT checkpoint
5. save raw doc multivectors under `data/colbert_vectors`
6. `MuveraEncoder`
7. `MuveraStore`
8. return real MUVERA index metadata

What happens on `/experimental/muvera/real/search`:

1. `text_routes.py`
2. `ExperimentalRealMuveraService`
3. encode query with official ColBERT
4. retrieve MUVERA candidates
5. load saved ColBERT multivectors
6. rerank with ColBERT MaxSim
7. compare against proxy MUVERA and dense/hybrid search
8. return candidate and reranked results

## When running `src.main_colpali:app`

Available paths:

- `/health`
- `/upload`
- `/search`
- `/answer`
- `/debug/rows`
- `/ingest/pdf`
- `/debug/pages`
- `/visual/embed-pages`
- `/visual/search`

What happens on `/ingest/pdf`:

1. `visual_routes.py`
2. `PageIndexingService`
3. `extract_pdf_markdown_and_images()`
4. `RetrievalStore.add_page_rows()`

What happens on `/visual/embed-pages`:

1. `visual_routes.py`
2. `VisualIndexingService`
3. lazy load `ColQwen2Service`
4. embed each stored page image
5. rewrite page table with real visual vectors

What happens on `/visual/search`:

1. `visual_routes.py`
2. `VisualIndexingService.search()`
3. embed query with `ColQwen2Service`
4. LanceDB page vector search
5. ColQwen-style scoring
6. return ranked page hits

---

## Practical Mental Model

If you want one simple way to think about this repo:

- `api/` exposes HTTP
- `services/` coordinates use cases
- `retrieval/` stores and searches evidence
- `ingest/` prepares documents
- `rerank/` improves ordering
- `main_colbert.py` is the text + official ColBERT app
- `experimental_muvera_service.py` is the proxy MUVERA experiment
- `experimental_real_muvera_service.py` is the real ColBERT-backed MUVERA experiment
- `main_colpali.py` is the text + visual app

If you are debugging a request, start from the matching route file, then follow the service it calls, then follow the retrieval or ingest modules underneath it.

Good. Here is a **practical implementation blueprint** for a PDF/document RAG product using **late interaction retrieval**, with **ColBERT / ColQwen-style reranking**, optional **MUVERA-style compression**, and **LangGraph** as the orchestration layer.

## 1) Recommended system design

### Core idea

Use a **two-stage retrieval pipeline**:

1. **Cheap candidate retrieval**

   * BM25, hybrid search, or compressed single-vector proxy
2. **Expensive late-interaction reranking**

   * ColBERT for text chunks
   * ColQwen/ColPali-style multi-vector scoring for page images

This is the production-friendly shape because LangGraph is best at orchestrating stateful retrieval workflows, while the actual late-interaction index/search should live in a dedicated retrieval service. LangGraph provides durable execution, persistence, and graph-based control flow; LanceDB explicitly supports multivector search for late-interaction models; Vespa explicitly supports ColBERT-style embeddings and discusses the memory tradeoff. ([LangChain Docs][1])

### Architecture

```text
                ┌──────────────────────┐
                │   Web / API Client   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   App API (FastAPI)  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   LangGraph Flow     │
                │ classify / rewrite   │
                │ retrieve / rerank    │
                │ answer / cite        │
                └───────┬───────┬──────┘
                        │       │
          ┌─────────────┘       └──────────────┐
          ▼                                    ▼
┌──────────────────────┐             ┌──────────────────────┐
│ Retrieval Service    │             │  LLM Answer Service  │
│                      │             │                      │
│ - BM25 / hybrid      │             │ - prompt assembly    │
│ - ANN / compressed   │             │ - grounded answer    │
│ - late reranker      │             │ - citation cleanup   │
└──────────┬───────────┘             └──────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────┐
│ Index Storage                                               │
│ - text chunks                                               │
│ - page images                                               │
│ - token/patch multivectors                                  │
│ - metadata: doc_id, page, section, OCR text, timestamps     │
└────────────────────────────────────────────────────────────┘
```

## 2) Ingestion flow

You should ingest **both text and visual evidence** for PDFs if your documents contain tables, forms, layouts, charts, or scans.

### Text path

```text
PDF -> extract text/OCR -> chunk passages -> ColBERT embeddings -> store multivectors
```

### Visual path

```text
PDF -> render each page to image -> ColQwen/ColPali embeddings -> store page multivectors
```

### Metadata to store

For each chunk/page:

* `doc_id`
* `page_number`
* `chunk_id`
* `section_title`
* `source_type` = `text_chunk` or `page_image`
* `raw_text`
* `ocr_text`
* `embedding_count`
* `created_at`
* optional `muvera_proxy_vector`

This matters because hybrid search and reranking usually need filters, citations, and page reconstruction. LanceDB supports vector, hybrid, and multivector search; Vespa supports embedding + ranking configurations for ColBERT-style retrieval. ([docs.lancedb.com][2])

## 3) Query-time flow

This is the flow I would actually deploy:

```text
user query
   ↓
query classifier
   ↓
query rewrite / expansion
   ↓
stage-1 retrieval
   ├─ BM25 / full text
   ├─ hybrid search
   └─ compressed proxy retrieval (MUVERA-style idea)
   ↓
top 100–300 candidates
   ↓
late interaction reranking
   ├─ ColBERT on text chunks
   └─ ColQwen on page images
   ↓
top 5–20 evidence items
   ↓
context packer
   ↓
LLM answer generation
   ↓
citations + logs + eval traces
```

### Why this is the right shape

LangChain recommends using LangChain for fast starts and LangGraph when you need custom orchestration; LangGraph also has first-class persistence/checkpointing for failure recovery and debugging. For retrieval-heavy products, this split is more robust than burying all retrieval logic inside one chain. ([LangChain Docs][3])

## 4) Where MUVERA fits

Treat MUVERA as a **candidate-generation accelerator**, not as the final relevance judge.

### Practical interpretation

At indexing time:

* you have many token/patch embeddings per document
* build a compressed or fixed-dimensional proxy representation

At query time:

* use the proxy representation to retrieve a shortlist quickly
* rerank shortlist with full late interaction

So your production flow becomes:

```text
multivector docs
   -> compressed proxy index
   -> fast shortlist retrieval
   -> full ColBERT/ColQwen rerank
```

That is the easiest way to get most of the scale benefit without giving up the detailed scoring that makes late interaction useful.

## 5) LangGraph pipeline design

Use LangGraph as the control plane.

### Suggested graph nodes

```text
START
  ↓
classify_query
  ↓
rewrite_query
  ↓
retrieve_candidates
  ↓
route_reranker
   ├─ rerank_text
   ├─ rerank_visual
   └─ rerank_hybrid_merge
  ↓
pack_context
  ↓
generate_answer
  ↓
validate_grounding
  ↓
format_citations
  ↓
END
```

### Recommended shared state

```python
from typing import TypedDict, List, Optional, Dict, Any

class RAGState(TypedDict, total=False):
    user_query: str
    rewritten_query: str
    query_type: str              # text / visual / mixed
    candidate_ids: List[str]
    text_hits: List[Dict[str, Any]]
    visual_hits: List[Dict[str, Any]]
    merged_hits: List[Dict[str, Any]]
    final_context: List[Dict[str, Any]]
    answer: str
    grounded: bool
    citations: List[Dict[str, Any]]
    debug: Dict[str, Any]
```

### Node responsibilities

**classify_query**

* decides whether query needs text retrieval, visual retrieval, or both
  Examples:
* “What does the contract say about termination?” → text
* “Find the page with the pricing table” → visual or mixed
* “Which page shows the red highlighted total due?” → visual

**rewrite_query**

* normalize the user query
* add synonyms/domain terms
* keep an audit copy for eval

**retrieve_candidates**

* run hybrid/BM25/multivector proxy search
* return top 100–300 ids

**route_reranker**

* if query is mostly textual, prioritize ColBERT
* if layout/table/visual language is present, prioritize ColQwen
* if mixed, run both then merge

**pack_context**

* reconstruct passages/pages
* deduplicate overlapping evidence
* keep page numbers and chunk offsets

**validate_grounding**

* answer should only use retrieved evidence
* if weak grounding, either abstain or trigger another retrieval pass

## 6) Example LangGraph skeleton

A minimal version would look like this:

```python
from langgraph.graph import StateGraph, END

def classify_query(state):
    q = state["user_query"].lower()
    visual_terms = ["page", "table", "chart", "figure", "layout", "scan", "image", "form"]
    state["query_type"] = "visual" if any(t in q for t in visual_terms) else "text"
    return state

def rewrite_query(state):
    state["rewritten_query"] = state["user_query"].strip()
    return state

def retrieve_candidates(state):
    # call your retrieval service here
    # hybrid / BM25 / compressed proxy index
    state["candidate_ids"] = ["doc1:p3", "doc4:p10", "doc7:c22"]
    return state

def rerank_text(state):
    # call ColBERT reranker
    state["text_hits"] = [
        {"id": "doc7:c22", "score": 0.92, "page": 5, "text": "Termination clause ..."}
    ]
    return state

def rerank_visual(state):
    # call ColQwen/ColPali reranker
    state["visual_hits"] = [
        {"id": "doc1:p3", "score": 0.89, "page": 3, "text": "Pricing table page"}
    ]
    return state

def merge_hits(state):
    merged = []
    merged.extend(state.get("text_hits", []))
    merged.extend(state.get("visual_hits", []))
    merged.sort(key=lambda x: x["score"], reverse=True)
    state["merged_hits"] = merged[:10]
    state["final_context"] = merged[:5]
    return state

def generate_answer(state):
    ctx = state.get("final_context", [])
    state["answer"] = f"Answer built from {len(ctx)} retrieved items."
    return state

def format_citations(state):
    state["citations"] = [
        {"doc_id": x["id"], "page": x.get("page")} for x in state.get("final_context", [])
    ]
    return state

graph = StateGraph(RAGState)
graph.add_node("classify_query", classify_query)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("retrieve_candidates", retrieve_candidates)
graph.add_node("rerank_text", rerank_text)
graph.add_node("rerank_visual", rerank_visual)
graph.add_node("merge_hits", merge_hits)
graph.add_node("generate_answer", generate_answer)
graph.add_node("format_citations", format_citations)

graph.set_entry_point("classify_query")
graph.add_edge("classify_query", "rewrite_query")
graph.add_edge("rewrite_query", "retrieve_candidates")
graph.add_edge("retrieve_candidates", "rerank_text")
graph.add_edge("rerank_text", "rerank_visual")
graph.add_edge("rerank_visual", "merge_hits")
graph.add_edge("merge_hits", "generate_answer")
graph.add_edge("generate_answer", "format_citations")
graph.add_edge("format_citations", END)

app = graph.compile()
```

LangGraph’s graph primitives, persistence, and durable execution are designed for exactly this kind of multi-step stateful workflow. ([LangChain Docs][4])

## 7) Tooling stack I would choose

### Easiest stack

* **FastAPI** for API layer
* **LangGraph** for orchestration
* **LanceDB** for multivector/hybrid search
* **ColBERT** for text reranking
* **ColQwen/ColPali-style model** for visual page reranking
* **LangSmith** for traces/evals

Why:

* LanceDB explicitly supports multivector search and hybrid/full-text search, which is a good fit for ColBERT-style retrieval. ([docs.lancedb.com][2])

### Heavier enterprise stack

* **Vespa** for retrieval/indexing/ranking
* **LangGraph** for workflow
* **LangSmith** for evals

Why:

* Vespa has explicit ColBERT support and stronger ranking/filtering infra, but is heavier operationally. Vespa also notes ColBERT’s memory cost and recommends compression/binarization for large-scale use. ([docs.vespa.ai][5])

## 8) Testing plan: how to prove it improved

Do this in layers.

### Layer 1: retrieval benchmark

Create a dataset like:

```json
{
  "query": "find the page with the pricing table for enterprise tier",
  "relevant_ids": ["contract_17:page_12"]
}
```

Collect 100–500 such examples.

### Metrics

Use:

* **Recall@k**
* **MRR**
* **nDCG**
* **Hit@k**

Measure these across:

1. BM25 only
2. single-vector dense retriever
3. hybrid retrieval
4. hybrid + ColBERT rerank
5. hybrid + ColQwen rerank
6. compressed proxy + late rerank

### Layer 2: answer benchmark

For each question, store:

* gold answer or reference facts
* relevant citation target
* must-have evidence ids

Then measure:

* answer correctness
* faithfulness / groundedness
* citation correctness
* context precision
* context recall

LangSmith documents offline and online evaluation flows for RAG, including dataset-based testing and production monitoring. ([LangChain Docs][6])

## 9) A/B matrix you should actually run

Use the same corpus and same test set.

```text
A. BM25
B. Dense single-vector
C. Hybrid
D. Hybrid + ColBERT rerank
E. Hybrid + visual rerank
F. Compressed proxy + full rerank
```

Track for each:

* Recall@20
* MRR@10
* answer groundedness
* median latency
* p95 latency
* storage size
* indexing time
* GPU cost per 1k queries

That gives you a real business answer, not just a research answer.

## 10) What usually improves

### When ColBERT helps

* long text chunks
* technical wording
* exact-but-contextual phrase matching
* domain-specific corpora

### When ColQwen helps

* tables
* forms
* invoices
* charts
* multi-column layouts
* poor OCR pages

### When MUVERA-style compression helps

* larger corpora
* lower latency targets
* cheaper first-stage retrieval
* keeping most of late-interaction quality without full-corpus rerank

## 11) Recommended rollout plan

### Phase 1

* Hybrid retrieval only
* Baseline eval set
* LangSmith tracing

### Phase 2

* Add ColBERT reranking
* Measure retrieval lift

### Phase 3

* Add visual page reranking for PDF-heavy queries
* Measure layout/table improvement

### Phase 4

* Add compressed proxy index
* Measure latency vs quality tradeoff

That rollout reduces risk and lets you attribute gains correctly.

## 12) My concrete recommendation for you

If you want something you can actually ship:

### For mostly text PDFs

* **Hybrid retrieval + ColBERT rerank**
* **LangGraph**
* **LanceDB**
* **LangSmith**

### For visually complex PDFs

* same as above, plus
* **page-image indexing**
* **ColQwen/ColPali visual reranker**

### For scale

* add
* **MUVERA-style compressed first-stage retrieval**
* keep full late interaction only for reranking shortlist

That is the cleanest path from prototype to production.

I can now turn this into either:

1. a **full code starter project structure**, or
2. a **diagram + API contract + database schema**.

[1]: https://docs.langchain.com/oss/python/langgraph/overview?utm_source=chatgpt.com "LangGraph overview - Docs by LangChain"
[2]: https://docs.lancedb.com/search/multivector-search?utm_source=chatgpt.com "Multivector Search"
[3]: https://docs.langchain.com/oss/python/langchain/overview?utm_source=chatgpt.com "LangChain overview - Docs by LangChain"
[4]: https://docs.langchain.com/oss/python/langgraph/graph-api?utm_source=chatgpt.com "Graph API overview - Docs by LangChain"
[5]: https://docs.vespa.ai/en/rag/embedding.html?utm_source=chatgpt.com "Embedding"
[6]: https://docs.langchain.com/langsmith/evaluate-rag-tutorial?utm_source=chatgpt.com "Evaluate a RAG application - Docs by LangChain"


===========
Perfect. We should build this in **layers** so you learn the retrieval system instead of pasting a giant repo and hoping it works.

The stack I’d use is:

* **uv** for project/env/package management, because it manages `pyproject.toml`, lockfiles, environments, and `uv run`/`uv sync` cleanly. ([docs.astral.sh][1])
* **LangGraph** for orchestration, because it is the lower-level runtime for stateful, long-running workflows rather than the retriever itself. ([LangChain Docs][2])
* **LanceDB** for storage, because it has documented **multivector search** support for late-interaction models like ColBERT and ColPali. ([LanceDB][3])
* **ColBERT** for text late-interaction reranking. The Stanford ColBERT repo is still the canonical reference implementation. ([GitHub][4])
* **ColQwen2** as the primary visual retriever for PDF page images, because its model card describes it as a Qwen2-VL-2B extension that produces **ColBERT-style multi-vector representations of text and images**. ColPali is a good alternative built on PaliGemma-3B. ([Hugging Face][5])
* **MUVERA-style candidate generation** as a pluggable first stage, because MUVERA reduces multi-vector retrieval to single-vector similarity search using fixed-dimensional encodings. ([arXiv][6])

The important design choice is this:

**do not start with full complexity on day 1.**
Start with a clean architecture where MUVERA, ColQwen2, and ColBERT each fit naturally, then turn them on one by one.

---

# Phase 0: what we are building

End-state flow:

```text
User query
  -> LangGraph router
  -> Stage 1 candidate retrieval
       - BM25 over extracted text
       - dense text retrieval
       - MUVERA/FDE proxy retrieval for multivectors
  -> Stage 2 reranking
       - ColBERT for text chunks
       - ColQwen2 for page-image candidates
  -> Context packer
  -> Answer generation
  -> Citations + metrics
```

For PDFs we will store **two views** of the same source:

1. **text view**
   extracted text / OCR / chunks

2. **visual page view**
   rendered page image + page-level multivectors

That gives you hybrid and visual retrieval without forcing every query through the expensive path.

---

# Phase 1: create the uv project

Start here.

```bash
mkdir multivector-rag
cd multivector-rag

uv init
```

`uv init` creates a project with `pyproject.toml`, `.python-version`, and starter files, and `uv run`/`uv sync` will create `.venv` and `uv.lock` automatically. ([docs.astral.sh][1])

Now add the base packages:

```bash
uv add fastapi uvicorn pydantic pydantic-settings
uv add langgraph langchain
uv add lancedb pyarrow tantivy
uv add pymupdf pillow
uv add rank-bm25 numpy scipy pandas
uv add transformers torch
uv add pytest httpx
```

A few notes:

* `uv run ...` runs inside the managed project environment and keeps it synced with the lockfile. ([docs.astral.sh][1])
* LanceDB supports multivector search, which is why we use it early instead of trying to retrofit a single-vector DB later. ([LanceDB][3])

Now make the folder structure:

```bash
mkdir -p src/app
mkdir -p src/core
mkdir -p src/ingest
mkdir -p src/retrieval
mkdir -p src/rerank
mkdir -p src/graph
mkdir -p src/models
mkdir -p src/api
mkdir -p data/raw
mkdir -p data/processed
mkdir -p tests
touch src/app/__init__.py
touch src/main.py
```

Use this layout:

```text
multivector-rag/
  pyproject.toml
  src/
    main.py
    app/
    core/
      config.py
      types.py
    ingest/
      pdf.py
      text.py
      images.py
      chunking.py
    retrieval/
      bm25.py
      dense.py
      hybrid.py
      muvera_proxy.py
      store.py
    rerank/
      colbert.py
      colqwen.py
      router.py
    graph/
      state.py
      workflow.py
    api/
      routes.py
  data/
    raw/
    processed/
  tests/
```

---

# Phase 2: wire the app skeleton first

Before retrieval, make sure the project runs.

## `src/main.py`

```python
from fastapi import FastAPI

app = FastAPI(title="Multivector RAG")

@app.get("/health")
def health():
    return {"status": "ok"}
```

Run it:

```bash
uv run uvicorn src.main:app --reload
```

That uses `uv run` so you don’t have to manually activate the environment. ([docs.astral.sh][1])

---

# Phase 3: ingest PDFs into text + page images

This is where learning starts to matter.

We want each PDF to become:

* extracted text
* page images
* chunk metadata

## `src/ingest/pdf.py`

```python
from pathlib import Path
import fitz  # PyMuPDF

def extract_pdf_text_and_images(pdf_path: str, out_dir: str) -> list[dict]:
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_path = out_dir / f"{pdf_path.stem}_page_{i+1}.png"
        pix.save(str(image_path))

        pages.append({
            "page_number": i + 1,
            "text": text,
            "image_path": str(image_path),
            "source_pdf": str(pdf_path),
        })

    return pages
```

## `src/ingest/chunking.py`

```python
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += max(1, chunk_size - overlap)
    return chunks
```

## Why do this first?

Because everything later depends on the schema. For each PDF page, you need a stable record like:

```python
{
    "doc_id": "contract_001",
    "page_number": 4,
    "page_text": "...",
    "page_image_path": "data/processed/contract_001_page_4.png",
    "chunks": [...]
}
```

Without that discipline, hybrid retrieval becomes messy.

---

# Phase 4: add a local store layer

We want one place that knows how to save and retrieve records.

## `src/retrieval/store.py`

```python
from pathlib import Path
import lancedb

class RetrievalStore:
    def __init__(self, uri: str = "data/lancedb"):
        Path(uri).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(uri)

    def get_or_create_text_table(self):
        schema = [
            {"name": "id", "type": "string"},
            {"name": "doc_id", "type": "string"},
            {"name": "page_number", "type": "int32"},
            {"name": "chunk_text", "type": "string"},
            {"name": "vector", "type": "list<float32>"},
        ]
        try:
            return self.db.open_table("text_chunks")
        except Exception:
            return self.db.create_table("text_chunks", data=[])

    def get_or_create_pages_table(self):
        try:
            return self.db.open_table("page_images")
        except Exception:
            return self.db.create_table("page_images", data=[])
```

You’ll improve the schema later. For now, just make the storage boundary explicit.

LanceDB’s multivector support is one reason it fits this project well. ([LanceDB][3])

---

# Phase 5: baseline retrieval before MUVERA

Do **not** jump straight into ColQwen2 + MUVERA. First build a baseline you can measure against.

## A. BM25

## `src/retrieval/bm25.py`

```python
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self):
        self.docs = []
        self.doc_ids = []
        self.index = None

    def build(self, rows: list[dict]):
        self.docs = [r["chunk_text"].split() for r in rows]
        self.doc_ids = [r["id"] for r in rows]
        self.index = BM25Okapi(self.docs)

    def search(self, query: str, top_k: int = 10):
        if self.index is None:
            return []
        scores = self.index.get_scores(query.split())
        ranked = sorted(
            zip(self.doc_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        return [{"id": doc_id, "score": float(score)} for doc_id, score in ranked]
```

## B. Dense retrieval

Use a small text embedder first. Keep this lightweight for learning. Later we can swap in a better model.

## `src/retrieval/dense.py`

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class DenseTextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            out = self.model(**encoded)
            cls = out.last_hidden_state[:, 0]
            cls = F.normalize(cls, p=2, dim=1)
        return cls.cpu().tolist()
```

## C. Hybrid merge

## `src/retrieval/hybrid.py`

```python
def reciprocal_rank_fusion(*rank_lists, k: int = 60):
    scores = {}
    for rank_list in rank_lists:
        for rank, item in enumerate(rank_list, start=1):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return [
        {"id": doc_id, "score": score}
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
```

This gives you your first working retrieval stack:

* BM25
* dense vectors
* RRF fusion

That baseline matters because later you need to prove ColBERT, ColQwen2, and MUVERA actually improved something.

---

# Phase 6: page-image indexing with ColQwen2

Now we add the visual retrieval path.

I would choose **ColQwen2 first**, not ColPali, because the Vidore model card describes ColQwen2 as a Qwen2-VL-2B-based visual retriever that produces ColBERT-style multivector representations for text and images. ColPali is also valid, but I’d keep it as your second experiment. ([Hugging Face][5])

## Smart approach

Use ColQwen2 for:

* page candidate retrieval
* page reranking for visually complex queries

Use ColBERT for:

* chunk reranking when query is mostly textual

That avoids wasting visual inference on ordinary text questions.

## `src/rerank/colqwen.py`

Start with an interface, not the full model logic:

```python
class ColQwenRetriever:
    def __init__(self, model_name: str = "vidore/colqwen2-v1.0"):
        self.model_name = model_name
        self.ready = False

    def load(self):
        # load processor/model here later
        self.ready = True

    def embed_page_image(self, image_path: str):
        raise NotImplementedError("Implement ColQwen2 image multivector embedding")

    def embed_query(self, query: str):
        raise NotImplementedError("Implement ColQwen2 query embedding")

    def score(self, query_embedding, page_embedding):
        raise NotImplementedError("Implement late interaction / MaxSim scoring")
```

Why do this instead of dropping in 200 lines of model code immediately?
Because you need the system boundary first:

* where embeddings are created
* where they are stored
* where scores are computed
* where candidates are reranked

That boundary stays stable even if you later switch from ColQwen2 to ColPali.

---

# Phase 7: text late-interaction with ColBERT

ColBERT is the text-side reranker. The Stanford repo still frames it as the canonical late-interaction implementation. ([GitHub][4])

Again, start with the interface.

## `src/rerank/colbert.py`

```python
class ColBERTReranker:
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.model_name = model_name
        self.ready = False

    def load(self):
        self.ready = True

    def embed_query(self, query: str):
        raise NotImplementedError

    def embed_document(self, text: str):
        raise NotImplementedError

    def maxsim_score(self, q_vectors, d_vectors) -> float:
        raise NotImplementedError

    def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        raise NotImplementedError
```

At this stage the lesson is architectural:

* **hybrid** finds candidates
* **ColBERT** reranks text candidates
* **ColQwen2** reranks page-image candidates

---

# Phase 8: MUVERA as a pluggable first-stage proxy

This part needs honesty.

The MUVERA paper and Google Research write-up clearly describe the idea: build fixed-dimensional encodings so multi-vector retrieval can use ordinary single-vector MIPS infrastructure. ([arXiv][6])

But in practice, I did **not** find a simple, official, one-line Python package in the sources I checked. I did find community implementations, including Python and Rust-accelerated versions, but I would treat those as experimental until you validate them in your stack. ([GitHub][7])

So the smart engineering move is:

## Build MUVERA behind an interface

## `src/retrieval/muvera_proxy.py`

```python
class MuveraProxyIndex:
    def __init__(self):
        self.rows = []

    def fit(self, rows: list[dict]):
        """
        rows:
          {
            "id": "...",
            "multivectors": [[...], [...], ...]
          }
        """
        self.rows = rows

    def encode_multivector_to_fde(self, multivectors):
        raise NotImplementedError("Plug in MUVERA FDE here")

    def search(self, query_multivectors, top_k: int = 50):
        raise NotImplementedError("Search over FDE proxy vectors here")
```

## Why this is the right order

Because your pipeline should work even before MUVERA is fully wired:

1. hybrid text retrieval works
2. visual page store works
3. reranker routing works
4. then MUVERA replaces or augments candidate generation

That way MUVERA is a performance upgrade, not a blocker.

---

# Phase 9: router logic for “smart approach”

This is the key product feature.

We do **not** always run both rerankers.

## `src/rerank/router.py`

```python
VISUAL_HINTS = {
    "table", "figure", "chart", "diagram", "layout", "page",
    "form", "invoice", "scan", "screenshot", "image"
}

def classify_query(query: str) -> str:
    q = query.lower()
    if any(word in q for word in VISUAL_HINTS):
        return "visual"
    return "text"

def choose_rerankers(query: str) -> list[str]:
    query_type = classify_query(query)
    if query_type == "visual":
        return ["colqwen", "colbert"]
    return ["colbert"]
```

Later you’ll make this smarter with an LLM or learned classifier. For now:

* mostly text query → ColBERT only
* visual/layout/table/page query → ColQwen2 first, ColBERT second

That is a solid production heuristic.

---

# Phase 10: LangGraph workflow

LangGraph is for orchestration, not for storing embeddings. Its docs position it as the low-level framework for stateful orchestration, durable execution, and streaming. ([LangChain Docs][2])

## `src/graph/state.py`

```python
from typing import TypedDict, Any

class GraphState(TypedDict, total=False):
    query: str
    query_type: str
    bm25_hits: list[dict]
    dense_hits: list[dict]
    hybrid_hits: list[dict]
    muvera_hits: list[dict]
    text_reranked: list[dict]
    visual_reranked: list[dict]
    final_hits: list[dict]
    answer: str
    debug: dict[str, Any]
```

## `src/graph/workflow.py`

```python
from langgraph.graph import StateGraph, END
from src.graph.state import GraphState
from src.rerank.router import classify_query

def classify_node(state: GraphState):
    state["query_type"] = classify_query(state["query"])
    return state

def retrieve_node(state: GraphState):
    # call bm25, dense, hybrid, muvera proxy here
    return state

def rerank_node(state: GraphState):
    # route to ColBERT / ColQwen2 based on query_type
    return state

def answer_node(state: GraphState):
    state["answer"] = "Stub answer"
    return state

graph = StateGraph(GraphState)
graph.add_node("classify", classify_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("rerank", rerank_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("classify")
graph.add_edge("classify", "retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "answer")
graph.add_edge("answer", END)

app_graph = graph.compile()
```

This is enough to make the orchestration layer real, even before the heavy retrieval code is complete.

---

# Phase 11: tests from the start

You said you want to learn while building. Then we should test every layer.

Add these first tests:

## `tests/test_chunking.py`

```python
from src.ingest.chunking import chunk_text

def test_chunk_text():
    text = "hello " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)
```

## `tests/test_router.py`

```python
from src.rerank.router import classify_query

def test_visual_query():
    assert classify_query("find the pricing table on page 2") == "visual"

def test_text_query():
    assert classify_query("what is the termination clause") == "text"
```

Run:

```bash
uv run pytest
```

---

# Phase 12: the build order I want you to follow

Here is the actual order.

## Milestone 1

Get this working:

* uv project
* FastAPI `/health`
* PDF extraction to text + page images
* tests

## Milestone 2

Add baseline retrieval:

* BM25
* dense text retrieval
* hybrid RRF
* one `/search` endpoint

## Milestone 3

Store page image records:

* page metadata
* page image path
* placeholder visual embeddings
* one `/ingest/pdf` endpoint

## Milestone 4

Add ColQwen2 embedding pipeline:

* query embedding
* page-image embedding
* page candidate scoring

## Milestone 5

Add ColBERT reranking:

* rerank top text chunks
* compare baseline vs reranked

## Milestone 6

Add MUVERA proxy stage:

* plug FDE/proxy encoding behind `MuveraProxyIndex`
* use it only for candidate generation
* keep ColBERT/ColQwen2 for final rerank

## Milestone 7

Add answer generation and citations:

* retrieve top evidence
* build prompt
* return cited answer

That sequence keeps the project usable at every step.

---

# Phase 13: what to do this week

I would make your first week look like this:

## Day 1

* initialize with uv
* create folders
* make FastAPI run
* write chunking and PDF extraction

## Day 2

* build BM25
* build dense retriever
* add hybrid fusion
* test on 2–3 sample PDFs

## Day 3

* add page-image indexing schema
* store image paths + page metadata
* expose `/ingest/pdf`

## Day 4

* add ColQwen2 wrapper interface
* add ColBERT wrapper interface
* implement router

## Day 5

* wire LangGraph around the flow
* add logs and debug payloads
* start evaluation dataset

That gets you a working skeleton without hiding the architecture.

---

# Phase 14: the first commands I want you to run

Run these exactly:

```bash
mkdir multivector-rag
cd multivector-rag
uv init

uv add fastapi uvicorn pydantic pydantic-settings
uv add langgraph langchain
uv add lancedb pyarrow tantivy
uv add pymupdf pillow
uv add rank-bm25 numpy scipy pandas
uv add transformers torch
uv add pytest httpx

mkdir -p src/app src/core src/ingest src/retrieval src/rerank src/graph src/models src/api
mkdir -p data/raw data/processed tests
touch src/main.py src/app/__init__.py
```

Then put this in `src/main.py`:

```python
from fastapi import FastAPI

app = FastAPI(title="Multivector RAG")

@app.get("/health")
def health():
    return {"status": "ok"}
```

Then run:

```bash
uv run uvicorn src.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/health
```

You should see:

```json
{"status":"ok"}
```

---

# My recommendation on model choice

Start with:

* **ColQwen2 first**
* **ColPali later as an experiment**
* **ColBERT for text reranking**
* **MUVERA only as proxy candidate generation**

Why:

* ColQwen2 is explicitly positioned as a visual retriever with ColBERT-style multivector output for text and images. ([Hugging Face][5])
* ColPali is also strong, but it’s better used as a comparison run once your system is stable. ([Hugging Face][8])
* MUVERA is best treated as a speed layer, not your final judge. ([arXiv][6])

---

# Important warning before you start

The biggest mistake in this kind of project is trying to implement:

* PDF ingestion
* OCR
* multivector storage
* ColQwen2
* ColBERT
* MUVERA
* LangGraph
* answer generation

all in one jump.

That usually ends in a broken repo with no clear failure point.

Build the interfaces first. Then fill them one by one.

Your immediate next move is: **finish Milestone 1 and Milestone 2 before touching ColQwen2 or MUVERA**.

In the next message, I can give you **Step 1 only** with the exact files and code for:

* `pyproject.toml`
* `src/main.py`
* `src/ingest/pdf.py`
* `src/ingest/chunking.py`
* `tests/test_chunking.py`

so you can build the foundation cleanly.

[1]: https://docs.astral.sh/uv/guides/projects/ "Working on projects | uv"
[2]: https://docs.langchain.com/oss/python/langgraph/overview "LangGraph overview - Docs by LangChain"
[3]: https://docs.lancedb.com/search/multivector-search?utm_source=chatgpt.com "Multivector Search"
[4]: https://github.com/stanford-futuredata/ColBERT?utm_source=chatgpt.com "ColBERT: state-of-the-art neural search ( ..."
[5]: https://huggingface.co/vidore/colqwen2-v1.0?utm_source=chatgpt.com "vidore/colqwen2-v1.0"
[6]: https://arxiv.org/abs/2405.19504?utm_source=chatgpt.com "MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings"
[7]: https://github.com/sionic-ai/muvera-py?utm_source=chatgpt.com "Python Implementation of MUVERA (Multi-Vector Retrieval ..."
[8]: https://huggingface.co/vidore/colpali-v1.3?utm_source=chatgpt.com "vidore/colpali-v1.3"

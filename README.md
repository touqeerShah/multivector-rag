# multivector-ragPhase 0: what we are building

End-state flow:

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

For PDFs we will store two views of the same source:

text view
extracted text / OCR / chunks

visual page view
rendered page image + page-level multivectors

That gives you hybrid and visual retrieval without forcing every query through the expensive path.

uv add fastapi uvicorn pydantic pydantic-settings
uv add langgraph langchain
uv add lancedb pyarrow tantivy
uv add pymupdf pillow
uv add rank-bm25 numpy scipy pandas
uv add transformers torch
uv add pytest httpx


uv run uvicorn src.main:app --reload



Here is the actual order.

Milestone 1

Get this working:

uv project

FastAPI /health

PDF extraction to text + page images

tests

Milestone 2

Add baseline retrieval:

BM25

dense text retrieval

hybrid RRF

one /search endpoint

Milestone 3

Store page image records:

page metadata

page image path

placeholder visual embeddings

one /ingest/pdf endpoint

Milestone 4

Add ColQwen2 embedding pipeline:

query embedding

page-image embedding

page candidate scoring

Milestone 5

Add ColBERT reranking:

rerank top text chunks

compare baseline vs reranked

Milestone 6

Add MUVERA proxy stage:

plug FDE/proxy encoding behind MuveraProxyIndex

use it only for candidate generation

keep ColBERT/ColQwen2 for final rerank

Milestone 7

Add answer generation and citations:

retrieve top evidence

build prompt

return cited answer

That sequence keeps the project usable at every step.
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from src.core.config import settings
from src.services.indexing import IndexingService
from src.services.search_service import SearchService
from src.retrieval.dense import DenseEmbedder
from src.services.page_indexing import PageIndexingService
from src.services.visual_indexing import VisualIndexingService
from src.rerank.colqwen import ColQwen2Service

router = APIRouter()
embedder = DenseEmbedder()
indexing_service = IndexingService(embedder)
search_service = SearchService(embedder)
page_indexing_service = PageIndexingService()

visual_indexing_service = VisualIndexingService()
colqwen_service = ColQwen2Service()

@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_file"
    suffix = Path(filename).suffix.lower()

    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(
            status_code=400, detail="Only .pdf, .txt, .md are supported"
        )

    save_path = settings.raw_path / filename
    with open(save_path, "wb") as f:
        f.write(await file.read())

    result = indexing_service.index_file(str(save_path))
    search_service.rebuild_bm25()

    return result


@router.get("/search")
def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=50),
):
    return search_service.search(query=q, top_k=top_k)


@router.get("/debug/rows")
def debug_rows(limit: int = 5):
    rows = search_service.store.all_rows()

    sample = []
    for row in rows[:limit]:
        sample.append(
            {
                "id": row.get("id"),
                "doc_id": row.get("doc_id"),
                "file_type": row.get("file_type"),
                "page_number": row.get("page_number"),
                "chunk_index": row.get("chunk_index"),
                "section_heading": row.get("section_heading"),
                "chunk_text_preview": (row.get("chunk_text") or "")[:200],
                "image_path": row.get("image_path"),
                "vector_len": (
                    len(row.get("vector", []))
                    if isinstance(row.get("vector"), list)
                    else None
                ),
            }
        )

    return {
        "count": len(rows),
        "sample": sample,
    }


@router.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    filename = file.filename or "uploaded.pdf"
    suffix = Path(filename).suffix.lower()

    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only .pdf is supported")

    save_path = settings.raw_path / filename
    with open(save_path, "wb") as f:
        f.write(await file.read())

    result = page_indexing_service.index_pdf_pages(str(save_path))
    return result


@router.get("/debug/pages")
def debug_pages(limit: int = 5):
    rows = page_indexing_service.store.all_page_rows()

    sample = []
    for row in rows[:limit]:
        sample.append(
            {
                "id": row.get("id"),
                "doc_id": row.get("doc_id"),
                "page_number": row.get("page_number"),
                "image_path": row.get("image_path"),
                "visual_status": row.get("visual_status"),
                "visual_vector_len": (
                    len(row.get("visual_vector", []))
                    if isinstance(row.get("visual_vector"), list)
                    else None
                ),
                "page_text_preview": (row.get("page_text_preview") or "")[:160],
            }
        )

    return {
        "count": len(rows),
        "sample": sample,
    }


@router.post("/visual/embed-pages")
def embed_visual_pages():
    return visual_indexing_service.index_existing_pages()


@router.get("/visual/search")
def visual_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
):
    query_vec = colqwen_service.embed_query(q)
    hits = visual_indexing_service.store.page_vector_search(query_vec, top_k=top_k)

    scored = colqwen_service.score_query_to_pages(q, hits)

    response = []
    for row in scored[:top_k]:
        response.append(
            {
                "id": row["id"],
                "doc_id": row["doc_id"],
                "page_number": row["page_number"],
                "image_path": row["image_path"],
                "visual_status": row["visual_status"],
                "visual_score": row["visual_score"],
                "page_text_preview": row.get("page_text_preview", ""),
            }
        )

    return {
        "query": q,
        "count": len(response),
        "results": response,
    }
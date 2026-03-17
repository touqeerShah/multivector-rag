from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from src.core.config import settings
from src.services.page_indexing import PageIndexingService
from src.services.visual_indexing import VisualIndexingService


page_indexing_service = PageIndexingService()
visual_indexing_service = VisualIndexingService()

router = APIRouter()


@router.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    filename = file.filename or "uploaded.pdf"
    suffix = Path(filename).suffix.lower()

    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only .pdf is supported")

    save_path = settings.raw_path / filename
    with open(save_path, "wb") as f:
        f.write(await file.read())

    return page_indexing_service.index_pdf_pages(str(save_path))


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
    return visual_indexing_service.search(query=q, top_k=top_k)

from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from src.core.config import settings
from src.services.indexing import IndexingService
from src.services.search_service import SearchService
from src.retrieval.dense import DenseEmbedder

router = APIRouter()
embedder = DenseEmbedder()
indexing_service = IndexingService(embedder)
search_service = SearchService(embedder)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_file"
    suffix = Path(filename).suffix.lower()

    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, .md are supported")

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
                "vector_len": len(row.get("vector", [])) if isinstance(row.get("vector"), list) else None,
            }
        )

    return {
        "count": len(rows),
        "sample": sample,
    }
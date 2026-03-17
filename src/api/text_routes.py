from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from src.core.config import settings
from src.retrieval.dense import DenseEmbedder
from src.services.indexing import IndexingService
from src.services.search_service import SearchService


_embedder = None
_indexing_service = None
_search_service = None
_experimental_text_indexing_service = None
_experimental_search_service = None


def _get_embedder():
    global _embedder

    if _embedder is None:
        _embedder = DenseEmbedder()

    return _embedder


def _get_indexing_service():
    global _indexing_service

    if _indexing_service is None:
        _indexing_service = IndexingService(_get_embedder())

    return _indexing_service


def _get_search_service():
    global _search_service

    if _search_service is None:
        _search_service = SearchService(_get_embedder())

    return _search_service


def _get_experimental_text_indexing_service():
    global _experimental_text_indexing_service

    if _experimental_text_indexing_service is None:
        from src.services.experimental_text_indexing import ExperimentalTextIndexingService

        _experimental_text_indexing_service = ExperimentalTextIndexingService()

    return _experimental_text_indexing_service


def _get_experimental_search_service():
    global _experimental_search_service

    if _experimental_search_service is None:
        from src.services.experimental_search_service import ExperimentalSearchService

        _experimental_search_service = ExperimentalSearchService(embedder=_get_embedder())

    return _experimental_search_service


def build_text_router(include_official_colbert: bool = False) -> APIRouter:
    router = APIRouter()

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

        indexing_service = _get_indexing_service()
        search_service = _get_search_service()

        result = indexing_service.index_file(str(save_path))
        search_service.rebuild_bm25()

        return result

    @router.get("/search")
    def search(
        q: str = Query(..., min_length=1),
        top_k: int = Query(10, ge=1, le=50),
    ):
        search_service = _get_search_service()
        return search_service.search(query=q, top_k=top_k)

    @router.get("/debug/rows")
    def debug_rows(limit: int = 5):
        search_service = _get_search_service()
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

    if include_official_colbert:
        from src.retrieval.colbert_service import ColBERTEnvironmentError

        @router.post("/experimental/colbert/reindex")
        def rebuild_colbert():
            try:
                return _get_experimental_text_indexing_service().rebuild_colbert_index(
                    overwrite=True
                )
            except ColBERTEnvironmentError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        @router.post("/experimental/colbert/reindex/background")
        def rebuild_colbert_in_background():
            try:
                return _get_experimental_text_indexing_service().start_rebuild_colbert_index(
                    overwrite=True
                )
            except ColBERTEnvironmentError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        @router.get("/experimental/colbert/reindex/status")
        def colbert_reindex_status():
            return _get_experimental_text_indexing_service().get_rebuild_status()

        @router.get("/experimental/search")
        def experimental_search(
            q: str = Query(..., min_length=1),
            top_k: int = Query(10, ge=1, le=50),
        ):
            try:
                return _get_experimental_search_service().search(
                    query=q, top_k=top_k
                )
            except ColBERTEnvironmentError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

    return router

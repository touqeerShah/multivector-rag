from __future__ import annotations

from typing import Dict, Any, List

from src.retrieval.store import RetrievalStore


class VisualIndexingService:
    def __init__(self):
        self.store = RetrievalStore()
        self._colqwen = None

    def _colqwen_service(self):
        if self._colqwen is None:
            from src.rerank.colqwen import ColQwen2Service

            self._colqwen = ColQwen2Service()

        return self._colqwen

    def index_existing_pages(self) -> Dict[str, Any]:
        pages = self.store.all_page_rows()
        updated_rows: List[Dict[str, Any]] = []
        colqwen = self._colqwen_service()

        for page in pages:
            if page.get("visual_status") == "embedded":
                continue

            image_path = page["image_path"]
            print(f"Embedding visual for page {page['id']} at {image_path}")
            visual_vector = colqwen.embed_page_image(image_path)
            print("visual vector dim:", len(visual_vector))
            updated_rows.append(
                {
                    "id": page["id"],
                    "doc_id": page["doc_id"],
                    "source_file": page["source_file"],
                    "page_number": page["page_number"],
                    "image_path": page["image_path"],
                    "markdown": page.get("markdown", ""),
                    "page_text_preview": page.get("page_text_preview", ""),
                    "visual_vector": visual_vector,
                    "visual_status": "embedded",
                }
            )

        if updated_rows:
            self._replace_page_table(updated_rows, pages)

        return {
            "status": "embedded",
            "updated_pages": len(updated_rows),
        }

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        colqwen = self._colqwen_service()
        query_vec = colqwen.embed_query(query)
        hits = self.store.page_vector_search(query_vec, top_k=top_k)
        scored = colqwen.score_query_to_pages(query, hits)

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
            "query": query,
            "count": len(response),
            "results": response,
        }

    def _replace_page_table(
        self,
        updated_rows: List[Dict[str, Any]],
        existing_rows: List[Dict[str, Any]],
    ) -> None:
        merged = {row["id"]: row for row in existing_rows}
        for row in updated_rows:
            merged[row["id"]] = row

        db = self.store.db
        name = self.store.page_table_name

        try:
            db.drop_table(name)
        except Exception:
            pass

        db.create_table(
            name,
            data=list(merged.values()),
            schema=self.store._page_schema(),
        )

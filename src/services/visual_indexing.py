from __future__ import annotations

from typing import Dict, Any, List

from src.retrieval.store import RetrievalStore
from src.rerank.colqwen import ColQwen2Service


class VisualIndexingService:
    def __init__(self):
        self.store = RetrievalStore()
        self.colqwen = ColQwen2Service()

    def index_existing_pages(self) -> Dict[str, Any]:
        pages = self.store.all_page_rows()
        updated_rows: List[Dict[str, Any]] = []

        for page in pages:
            if page.get("visual_status") == "embedded":
                continue

            image_path = page["image_path"]
            print(f"Embedding visual for page {page['id']} at {image_path}")
            visual_vector = self.colqwen.embed_page_image(image_path)

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
            # easiest dev-time approach: recreate page table if you prefer
            # but for now append updated versions as new rows is wrong.
            # so use delete/recreate pattern during milestone 4.
            self._replace_page_table(updated_rows, pages)

        return {
            "status": "embedded",
            "updated_pages": len(updated_rows),
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

        db.create_table(name, data=list(merged.values()), schema=self.store._page_schema())
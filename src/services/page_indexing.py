from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
from uuid import uuid4

from src.ingest.pdf import extract_pdf_markdown_and_images
from src.retrieval.store import RetrievalStore


class PageIndexingService:
    def __init__(self):
        self.store = RetrievalStore()

    def index_pdf_pages(self, pdf_path: str) -> Dict[str, Any]:
        pdf_file = Path(pdf_path)
        pages = extract_pdf_markdown_and_images(
            pdf_path=str(pdf_file),
            output_dir="data/processed",
        )

        rows = self._build_page_rows(
            doc_id=pdf_file.stem,
            source_file=str(pdf_file),
            pages=pages,
        )
        self.store.add_page_rows(rows)

        return {
            "status": "indexed",
            "file": str(pdf_file),
            "pages_indexed": len(rows),
        }

    def _build_page_rows(
        self,
        doc_id: str,
        source_file: str,
        pages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for page in pages:
            preview = (page.get("text") or page.get("markdown") or "")[:400]

            rows.append(
                {
                    "id": f"{doc_id}-page-{page['page_number']}-{uuid4().hex[:8]}",
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "page_number": page["page_number"],
                    "image_path": page["image_path"],
                    "markdown": page.get("markdown", ""),
                    "page_text_preview": preview,
                    "visual_vector": [0.0] * 128,
                    "visual_status": "placeholder",
                }
            )

        return rows

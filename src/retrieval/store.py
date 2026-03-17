from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import lancedb
import pyarrow as pa

from src.core.config import settings


TEXT_VECTOR_DIM = 384
VISUAL_VECTOR_DIM = 128  # change this after checking real ColQwen2 pooled vector length


class RetrievalStore:
    def __init__(
        self,
        uri: str | None = None,
        text_table_name: str | None = None,
        page_table_name: str | None = None,
    ):
        self.uri = uri or str(settings.lancedb_path)
        self.text_table_name = text_table_name or settings.text_table
        self.page_table_name = page_table_name or "page_images"

        Path(self.uri).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(self.uri)

        self._get_or_create_text_table()
        self._get_or_create_page_table()

    def _text_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("source_file", pa.string()),
                pa.field("file_type", pa.string()),
                pa.field("page_number", pa.int32()),
                pa.field("chunk_index", pa.int32()),
                pa.field("section_heading", pa.string()),
                pa.field("section_level", pa.int32()),
                pa.field("chunk_text", pa.string()),
                pa.field("image_path", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), TEXT_VECTOR_DIM)),
            ]
        )

    def _page_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("source_file", pa.string()),
                pa.field("page_number", pa.int32()),
                pa.field("image_path", pa.string()),
                pa.field("markdown", pa.string()),
                pa.field("page_text_preview", pa.string()),
                pa.field("visual_vector", pa.list_(pa.float32(), VISUAL_VECTOR_DIM)),
                pa.field("visual_status", pa.string()),
            ]
        )

    def _get_or_create_text_table(self):
        try:
            return self.db.open_table(self.text_table_name)
        except Exception:
            return self.db.create_table(
                self.text_table_name,
                data=[
                    {
                        "id": "init",
                        "doc_id": "init",
                        "source_file": "init",
                        "file_type": "init",
                        "page_number": 0,
                        "chunk_index": 0,
                        "section_heading": "",
                        "section_level": 0,
                        "chunk_text": "init",
                        "image_path": "",
                        "vector": [0.0] * TEXT_VECTOR_DIM,
                    }
                ],
                schema=self._text_schema(),
            )

    def _get_or_create_page_table(self):
        try:
            return self.db.open_table(self.page_table_name)
        except Exception:
            return self.db.create_table(
                self.page_table_name,
                data=[
                    {
                        "id": "init",
                        "doc_id": "init",
                        "source_file": "init",
                        "page_number": 0,
                        "image_path": "",
                        "markdown": "",
                        "page_text_preview": "",
                        "visual_vector": [0.0] * VISUAL_VECTOR_DIM,
                        "visual_status": "seed",
                    }
                ],
                schema=self._page_schema(),
            )

    def _text_table(self):
        return self.db.open_table(self.text_table_name)

    def _page_table(self):
        return self.db.open_table(self.page_table_name)

    def add_text_rows(self, rows: List[Dict[str, Any]]) -> None:
        cleaned_rows = [r for r in rows if r.get("id") != "init"]
        if cleaned_rows:
            self._text_table().add(cleaned_rows)

    def add_rows(self, rows: List[Dict[str, Any]]) -> None:
        self.add_text_rows(rows)

    def add_page_rows(self, rows: List[Dict[str, Any]]) -> None:
        cleaned_rows = [r for r in rows if r.get("id") != "init"]
        if cleaned_rows:
            self._page_table().add(cleaned_rows)

    def all_text_rows(self) -> List[Dict[str, Any]]:
        rows = self._text_table().to_pandas().to_dict(orient="records")
        return self._clean_rows(rows)

    def all_rows(self) -> List[Dict[str, Any]]:
        return self.all_text_rows()

    def all_page_rows(self) -> List[Dict[str, Any]]:
        rows = self._page_table().to_pandas().to_dict(orient="records")
        return self._clean_rows(rows)

    def text_vector_search(
        self, query_vector: List[float], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        results = self._text_table().search(query_vector).limit(top_k).to_list()
        return [r for r in results if r.get("id") != "init"]

    def page_vector_search(
        self, query_vector: List[float], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        results = (
            self._page_table()
            .search(query_vector, vector_column_name="visual_vector")
            .limit(top_k)
            .to_list()
        )
        return [r for r in results if r.get("id") != "init"]

    def _clean_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clean_rows: List[Dict[str, Any]] = []

        for row in rows:
            if row.get("id") == "init":
                continue

            clean_row: Dict[str, Any] = {}
            for key, value in row.items():
                if hasattr(value, "item"):
                    try:
                        value = value.item()
                    except Exception:
                        pass

                if hasattr(value, "tolist"):
                    try:
                        value = value.tolist()
                    except Exception:
                        pass

                if value is None:
                    value = ""

                clean_row[key] = value

            clean_rows.append(clean_row)

        return clean_rows

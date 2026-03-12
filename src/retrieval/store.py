from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import lancedb
import pyarrow as pa

from src.core.config import settings


class RetrievalStore:
    def __init__(self, uri: str | None = None, table_name: str | None = None):
        self.uri = uri or str(settings.lancedb_path)
        self.table_name = table_name or settings.text_table

        Path(self.uri).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(self.uri)
        self._get_or_create_table()

    def _schema(self) -> pa.Schema:
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
                pa.field("vector", pa.list_(pa.float32(), 384)),
            ]
        )

    def _empty_seed_row(self) -> List[Dict[str, Any]]:
        return [
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
                "vector": [0.0] * 384,
            }
        ]

    def _get_or_create_table(self):
        try:
            return self.db.open_table(self.table_name)
        except Exception:
            return self.db.create_table(
                self.table_name,
                data=self._empty_seed_row(),
                schema=self._schema(),
            )

    def _table(self):
        # Always reopen to avoid stale handles
        return self.db.open_table(self.table_name)

    def add_rows(self, rows: List[Dict[str, Any]]) -> None:
        cleaned_rows = [r for r in rows if r.get("id") != "init"]
        if cleaned_rows:
            self._table().add(cleaned_rows)

    def all_rows(self) -> List[Dict[str, Any]]:
        rows = self._table().to_pandas().to_dict(orient="records")

        clean_rows: List[Dict[str, Any]] = []
        for row in rows:
            if row.get("id") == "init":
                continue

            clean_row: Dict[str, Any] = {}
            for key, value in row.items():
                # numpy / pandas scalar -> python scalar
                if hasattr(value, "item"):
                    try:
                        value = value.item()
                    except Exception:
                        pass

                # ndarray / array-like -> list
                if hasattr(value, "tolist"):
                    try:
                        value = value.tolist()
                    except Exception:
                        pass

                # normalize null-ish values
                if value is None:
                    value = ""

                clean_row[key] = value

            clean_rows.append(clean_row)

        return clean_rows
    def vector_search(
        self, query_vector: List[float], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        results = self._table().search(query_vector).limit(top_k).to_list()
        return [r for r in results if r.get("id") != "init"]
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import json


class CollectionExporter:
    def __init__(self, base_dir: str = "data/colbert"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def export_collection_tsv(
        self,
        rows: List[Dict[str, Any]],
        filename: str = "collection.tsv",
        mapping_filename: str = "pid_mapping.json",
    ) -> dict[str, str]:
        output_path = self.base_dir / filename
        mapping_path = self.base_dir / mapping_filename

        pid_to_chunk_id: dict[str, str] = {}

        with output_path.open("w", encoding="utf-8") as f:
            for pid, row in enumerate(rows):
                real_id = row["id"]
                text = " ".join((row.get("chunk_text") or "").split())

                f.write(f"{pid}\t{text}\n")
                pid_to_chunk_id[str(pid)] = real_id

        mapping_path.write_text(
            json.dumps(pid_to_chunk_id, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "collection_tsv": str(output_path),
            "pid_mapping_json": str(mapping_path),
        }
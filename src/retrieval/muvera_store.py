from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any
import json
import numpy as np


class MuveraStore:
    def __init__(self, base_dir: str = "data/muvera"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_vectors_path = self.base_dir / "doc_fdes.npy"
        self.index_ids_path = self.base_dir / "doc_ids.json"

    def save_index(self, doc_ids: List[str], vectors: np.ndarray) -> None:
        np.save(self.index_vectors_path, vectors.astype(np.float32))
        self.index_ids_path.write_text(json.dumps(doc_ids), encoding="utf-8")

    def load_index(self) -> tuple[List[str], np.ndarray]:
        doc_ids = json.loads(self.index_ids_path.read_text(encoding="utf-8"))
        vectors = np.load(self.index_vectors_path)
        return doc_ids, vectors

    def search(self, query_fde: np.ndarray, top_k: int = 50) -> List[Dict[str, Any]]:
        doc_ids, doc_vectors = self.load_index()

        scores = doc_vectors @ query_fde.astype(np.float32)
        top_idx = np.argsort(-scores)[:top_k]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            results.append(
                {
                    "id": doc_ids[int(idx)],
                    "rank": rank,
                    "muvera_score": float(scores[int(idx)]),
                }
            )
        return results
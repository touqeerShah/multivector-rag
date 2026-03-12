from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.index: BM25Okapi | None = None

    def build(self, rows: List[Dict[str, Any]]) -> None:
        valid_rows = [r for r in rows if r.get("chunk_text") and r.get("id") != "init"]
        self.rows = valid_rows
        self.corpus_tokens = [r["chunk_text"].lower().split() for r in valid_rows]
        self.index = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.index is None:
            return []

        scores = self.index.get_scores(query.lower().split())
        ranked = sorted(
            zip(self.rows, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [
            {
                **row,
                "bm25_score": float(score),
            }
            for row, score in ranked
        ]
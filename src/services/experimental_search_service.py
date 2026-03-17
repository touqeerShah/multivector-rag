from __future__ import annotations

from typing import Dict, Any

from src.retrieval.store import RetrievalStore
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.colbert_service import OfficialColBERTService


class ExperimentalSearchService:
    def __init__(self, embedder):
        self.store = RetrievalStore()
        self.embedder = embedder
        self.bm25 = BM25Index()
        self.colbert = OfficialColBERTService()

    def rebuild_bm25(self):
        rows = self.store.all_text_rows()
        self.bm25.build(rows)

    def _public_hits(self, hits):
        public_hits = []
        for hit in hits:
            public_hit = {
                key: value
                for key, value in hit.items()
                if key not in {"vector", "visual_vector"}
            }
            public_hits.append(public_hit)
        return public_hits

    def _join_hits_with_metadata(self, hits):
        row_map = {row["id"]: row for row in self.store.all_text_rows()}
        joined = []
        for hit in hits:
            row = row_map.get(hit["id"])
            if row:
                joined.append({**row, **hit})
        return joined

    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        self.rebuild_bm25()

        bm25_hits = self.bm25.search(query, top_k=top_k)

        query_vector = self.embedder.embed_query(query)
        dense_hits = self.store.text_vector_search(query_vector, top_k=top_k)

        hybrid_hits = reciprocal_rank_fusion(bm25_hits, dense_hits)

        colbert_hits = self.colbert.search(query=query, top_k=top_k)
        colbert_joined = self._join_hits_with_metadata(colbert_hits)

        return {
            "query": query,
            "counts": {
                "bm25": len(bm25_hits),
                "dense": len(dense_hits),
                "hybrid": len(hybrid_hits[:top_k]),
                "colbert": len(colbert_joined),
            },
            "bm25": self._public_hits(bm25_hits),
            "dense": self._public_hits(dense_hits),
            "hybrid": self._public_hits(hybrid_hits[:top_k]),
            "colbert": self._public_hits(colbert_joined),
        }

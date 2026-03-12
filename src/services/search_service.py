from __future__ import annotations

from typing import Dict, Any

from src.retrieval.store import RetrievalStore
# from src.retrieval.embedder import DenseEmbedder
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import reciprocal_rank_fusion


class SearchService:
    def __init__(self,embedder):
        self.store = RetrievalStore()
        self.embedder = embedder
        self.bm25 = BM25Index()

    def rebuild_bm25(self):
        rows = self.store.all_rows()
        self.bm25.build(rows)

    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        self.rebuild_bm25()

        bm25_hits = self.bm25.search(query, top_k=top_k)

        query_vector = self.embedder.embed_query(query)
        dense_hits = self.store.vector_search(query_vector, top_k=top_k)

        hybrid_hits = reciprocal_rank_fusion(bm25_hits, dense_hits)

        return {
            "query": query,
            "counts": {
                "bm25": len(bm25_hits),
                "dense": len(dense_hits),
                "hybrid": len(hybrid_hits[:top_k]),
            },
            "bm25": bm25_hits,
            "dense": dense_hits,
            "hybrid": hybrid_hits[:top_k],
        }
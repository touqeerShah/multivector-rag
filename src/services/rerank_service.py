from __future__ import annotations

from typing import List, Dict, Any

from src.rerank.colbert_reranker import ColBERTReranker


class RerankService:
    def __init__(self):
        self.colbert = ColBERTReranker()

    def rerank_text_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        return self.colbert.rerank(query=query, docs=candidates, top_k=top_k)
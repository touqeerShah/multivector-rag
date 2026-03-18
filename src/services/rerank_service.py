from __future__ import annotations

from typing import List, Dict, Any

from src.retrieval.quality import lexical_overlap_count
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
        gated_candidates = []
        fallback_candidates = []

        for candidate in candidates:
            overlap = lexical_overlap_count(
                query=query,
                text=candidate.get("chunk_text", ""),
                heading=candidate.get("section_heading", ""),
            )
            enriched = {
                **candidate,
                "lexical_overlap": overlap,
            }
            fallback_candidates.append(enriched)
            if overlap > 0:
                gated_candidates.append(enriched)

        docs = gated_candidates or fallback_candidates
        return self.colbert.rerank(query=query, docs=docs, top_k=top_k)

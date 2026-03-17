from __future__ import annotations

from typing import List, Dict, Any
import math


class ColBERTReranker:
    """
    Milestone 5 starter reranker.

    This is a placeholder scoring layer that mimics a reranking stage.
    It is not true ColBERT late interaction yet.

    Why start here:
    - keeps your pipeline shape correct
    - lets you compare baseline vs reranked
    - easy to replace later with real ColBERT scoring
    """

    def __init__(self):
        self.ready = True

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        query_terms = self._normalize(query).split()

        scored: List[Dict[str, Any]] = []
        for doc in docs:
            text = self._normalize(doc.get("chunk_text", ""))
            section_heading = self._normalize(doc.get("section_heading", ""))

            lexical_hits = sum(text.count(term) for term in query_terms)
            heading_hits = sum(section_heading.count(term) for term in query_terms)

            length_penalty = math.log(len(text) + 10)

            score = (
                (2.0 * heading_hits)
                + (1.0 * lexical_hits)
            ) / max(length_penalty, 1.0)

            enriched = {
                **doc,
                "colbert_score": float(score),
            }
            scored.append(enriched)

        scored.sort(key=lambda x: x["colbert_score"], reverse=True)
        return scored[:top_k]

    def _normalize(self, text: str) -> str:
        return " ".join((text or "").lower().split())
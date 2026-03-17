from __future__ import annotations

import re
from typing import Any, Dict, List


class AnswerService:
    def __init__(self, search_service):
        self.search_service = search_service

    def answer(
        self,
        query: str,
        top_k: int = 10,
        evidence_k: int = 3,
    ) -> Dict[str, Any]:
        search_result = self.search_service.search(query=query, top_k=top_k)
        evidence = self._select_evidence(search_result, evidence_k=evidence_k)
        prompt = self.build_prompt(query=query, evidence=evidence)
        answer_text = self._generate_cited_answer(query=query, evidence=evidence)

        return {
            "query": query,
            "answer": answer_text,
            "citations": self._build_citations(evidence),
            "prompt": prompt,
            "evidence": evidence,
            "retrieval": search_result,
        }

    def _select_evidence(
        self,
        search_result: Dict[str, Any],
        evidence_k: int,
    ) -> List[Dict[str, Any]]:
        candidates = search_result.get("reranked") or search_result.get("hybrid") or []
        evidence: List[Dict[str, Any]] = []

        for idx, row in enumerate(candidates[:evidence_k], start=1):
            evidence.append(
                {
                    "citation": idx,
                    "id": row.get("id", ""),
                    "doc_id": row.get("doc_id", ""),
                    "source_file": row.get("source_file", ""),
                    "page_number": row.get("page_number"),
                    "section_heading": row.get("section_heading", ""),
                    "chunk_text": row.get("chunk_text", ""),
                    "score": row.get("colbert_score", row.get("_score", 0.0)),
                }
            )

        return evidence

    def build_prompt(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        context_blocks = []
        for item in evidence:
            heading = item["section_heading"] or "No heading"
            page = item["page_number"] if item["page_number"] not in ("", None) else "?"
            context_blocks.append(
                "\n".join(
                    [
                        f"[{item['citation']}] doc_id={item['doc_id']} page={page}",
                        f"heading={heading}",
                        item["chunk_text"],
                    ]
                )
            )

        joined_context = "\n\n".join(context_blocks) if context_blocks else "No evidence found."
        return "\n".join(
            [
                "You are answering from retrieved document evidence only.",
                "Use bracketed citations like [1] tied to the evidence blocks.",
                "If the evidence is insufficient, say so explicitly.",
                f"Question: {query}",
                "",
                "Evidence:",
                joined_context,
                "",
                "Answer:",
            ]
        )

    def _generate_cited_answer(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return "I could not find enough indexed evidence to answer that question."

        query_terms = set(self._normalize(query).split())
        chosen_sentences: List[str] = []

        for item in evidence:
            sentence = self._best_sentence(item["chunk_text"], query_terms)
            if not sentence:
                continue

            cited = f"{sentence} [{item['citation']}]"
            if cited not in chosen_sentences:
                chosen_sentences.append(cited)

        if not chosen_sentences:
            first = evidence[0]
            fallback = self._truncate(first["chunk_text"], 220)
            return f"{fallback} [{first['citation']}]"

        return " ".join(chosen_sentences[:3])

    def _build_citations(self, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations = []
        for item in evidence:
            citations.append(
                {
                    "citation": item["citation"],
                    "id": item["id"],
                    "doc_id": item["doc_id"],
                    "source_file": item["source_file"],
                    "page_number": item["page_number"],
                    "section_heading": item["section_heading"],
                    "snippet": self._truncate(item["chunk_text"], 240),
                }
            )
        return citations

    def _best_sentence(self, text: str, query_terms: set[str]) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return ""

        def score(sentence: str) -> tuple[int, int]:
            normalized = self._normalize(sentence)
            terms = normalized.split()
            overlap = sum(1 for term in terms if term in query_terms)
            return overlap, len(terms)

        best = max(sentences, key=score)
        return self._truncate(best.strip(), 220)

    def _split_sentences(self, text: str) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return []
        return [part for part in re.split(r"(?<=[.!?])\s+", cleaned) if part]

    def _truncate(self, text: str, max_chars: int) -> str:
        text = (text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).strip()

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np

from src.retrieval.muvera_encoder import MuveraEncoder
from src.retrieval.muvera_store import MuveraStore
from src.retrieval.store import RetrievalStore


class ExperimentalMuveraService:
    def __init__(
        self,
        embedder,
        search_service,
        store: RetrievalStore | None = None,
        muvera_encoder: MuveraEncoder | None = None,
        muvera_store: MuveraStore | None = None,
    ):
        self.embedder = embedder
        self.search_service = search_service
        self.store = store or RetrievalStore()
        self._muvera_encoder = muvera_encoder
        self.muvera_store = muvera_store or MuveraStore()

    def rebuild_index(
        self,
        max_subvectors_per_doc: int = 8,
    ) -> Dict[str, Any]:
        rows = self.store.all_text_rows()
        doc_ids: List[str] = []
        fdes: List[np.ndarray] = []
        subvectors_per_doc: List[int] = []

        for row in rows:
            multivector = self._text_to_multivector(
                row.get("chunk_text", ""),
                max_subvectors=max_subvectors_per_doc,
            )
            if multivector.size == 0:
                continue

            doc_ids.append(row["id"])
            subvectors_per_doc.append(int(multivector.shape[0]))
            fdes.append(self._encoder().encode_document_multivectors(multivector))

        if not doc_ids:
            self.muvera_store.save_index([], np.zeros((0, self._encoder().output_dim()), dtype=np.float32))
            return {
                "status": "indexed",
                "indexed_docs": 0,
                "proxy_vector_dim": self._encoder().output_dim(),
                "avg_subvectors_per_doc": 0.0,
                "notes": [
                    "No text rows were available, so the MUVERA proxy index is empty."
                ],
            }

        vectors = np.stack(fdes).astype(np.float32)
        self.muvera_store.save_index(doc_ids, vectors)

        return {
            "status": "indexed",
            "indexed_docs": len(doc_ids),
            "proxy_vector_dim": int(vectors.shape[1]),
            "avg_subvectors_per_doc": float(sum(subvectors_per_doc) / len(subvectors_per_doc)),
            "max_subvectors_per_doc": max_subvectors_per_doc,
            "notes": [
                "This is an experimental MUVERA proxy index built from dense mini-span embeddings.",
                "It reduces variable-length per-document subvectors into one fixed-size vector for candidate retrieval.",
            ],
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
        max_query_subvectors: int = 6,
        comparison_top_k: int | None = None,
    ) -> Dict[str, Any]:
        comparison_top_k = comparison_top_k or top_k
        query_multivector = self._text_to_multivector(
            query,
            max_subvectors=max_query_subvectors,
        )
        if query_multivector.size == 0:
            return {
                "query": query,
                "mode": "experimental_muvera_proxy",
                "counts": {"muvera": 0, "dense": 0, "hybrid": 0},
                "muvera": [],
                "dense": [],
                "hybrid": [],
                "notes": ["The query did not produce any MUVERA proxy subvectors."],
            }

        query_fde = self._encoder().encode_query_multivectors(query_multivector)
        hits = self.muvera_store.search(query_fde, top_k=top_k)
        muvera_joined = self._join_hits_with_metadata(hits)

        baseline = self.search_service.search(query=query, top_k=comparison_top_k)
        dense = baseline.get("dense", [])
        hybrid = baseline.get("hybrid", [])

        muvera_ids = {row["id"] for row in muvera_joined}
        dense_ids = {row["id"] for row in dense}
        hybrid_ids = {row["id"] for row in hybrid}

        return {
            "query": query,
            "mode": "experimental_muvera_proxy",
            "counts": {
                "muvera": len(muvera_joined),
                "dense": len(dense),
                "hybrid": len(hybrid),
                "overlap_dense": len(muvera_ids & dense_ids),
                "overlap_hybrid": len(muvera_ids & hybrid_ids),
            },
            "muvera_config": {
                "proxy_vector_dim": self._encoder().output_dim(),
                "query_subvectors": int(query_multivector.shape[0]),
                "max_query_subvectors": max_query_subvectors,
            },
            "muvera": muvera_joined,
            "dense": dense,
            "hybrid": hybrid,
            "notes": [
                "MUVERA here is a proxy experiment over dense mini-span embeddings, not full ColBERT token vectors.",
                "Use this endpoint to compare candidate ordering against the standard /search pipeline.",
            ],
        }

    def _encoder(self) -> MuveraEncoder:
        if self._muvera_encoder is None:
            self._muvera_encoder = MuveraEncoder(dimension=self._embedding_dim())
        return self._muvera_encoder

    def _embedding_dim(self) -> int:
        sample = self.embedder.embed_texts(["dimension probe"])
        return len(sample[0]) if sample else 0

    def _join_hits_with_metadata(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        row_map = {row["id"]: row for row in self.store.all_text_rows()}
        joined = []

        for hit in hits:
            row = row_map.get(hit["id"])
            if not row:
                continue

            joined.append(
                {
                    key: value
                    for key, value in {**row, **hit}.items()
                    if key not in {"vector", "visual_vector"}
                }
            )

        return joined

    def _text_to_multivector(
        self,
        text: str,
        max_subvectors: int,
    ) -> np.ndarray:
        spans = self._segment_text(text, max_segments=max_subvectors)
        if not spans:
            return np.zeros((0, self._embedding_dim()), dtype=np.float32)

        vectors = self.embedder.embed_texts(spans)
        return np.asarray(vectors, dtype=np.float32)

    def _segment_text(self, text: str, max_segments: int) -> List[str]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return []

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", normalized)
            if sentence.strip()
        ]
        if not sentences:
            sentences = [normalized]

        spans: List[str] = []
        for sentence in sentences:
            tokens = sentence.split()
            if len(tokens) <= 24:
                spans.append(sentence)
            else:
                step = 16
                for start in range(0, len(tokens), step):
                    spans.append(" ".join(tokens[start : start + 24]))

        return spans[:max_segments]

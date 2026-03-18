from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

from src.retrieval.colbert_service import ensure_colbert_runtime_compatible
from src.retrieval.muvera_encoder import MuveraEncoder
from src.retrieval.muvera_store import MuveraStore
from src.retrieval.quality import content_terms, filter_retrievable_rows, lexical_overlap_count
from src.retrieval.store import RetrievalStore


DEFINITIONAL_PREFIXES = (
    "what is",
    "what are",
    "define",
    "definition of",
    "explain",
)


class ExperimentalRealMuveraService:
    def __init__(
        self,
        search_service,
        proxy_muvera_service,
        store: RetrievalStore | None = None,
        vector_dir: str = "data/colbert_vectors",
        checkpoint_name: str = "colbert-ir/colbertv2.0",
        checkpoint: Checkpoint | None = None,
        muvera_encoder: MuveraEncoder | None = None,
        muvera_store: MuveraStore | None = None,
    ):
        self.search_service = search_service
        self.proxy_muvera_service = proxy_muvera_service
        self.store = store or RetrievalStore()
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = checkpoint_name
        self._checkpoint = checkpoint
        self._muvera_encoder = muvera_encoder
        self.muvera_store = muvera_store or MuveraStore(base_dir="data/muvera_real")

    def rebuild_index(
        self,
        top_docs: int | None = None,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        rows = filter_retrievable_rows(self.store.all_text_rows())
        if top_docs is not None:
            rows = rows[:top_docs]

        texts = [row.get("chunk_text", "") for row in rows if row.get("chunk_text")]
        valid_rows = [row for row in rows if row.get("chunk_text")]

        if not valid_rows:
            self.muvera_store.save_index(
                [],
                np.zeros((0, self._encoder().output_dim()), dtype=np.float32),
            )
            return {
                "status": "indexed",
                "indexed_docs": 0,
                "vector_dir": str(self.vector_dir),
                "muvera_vector_dim": self._encoder().output_dim(),
                "notes": [
                    "No text rows were available, so the real ColBERT-backed MUVERA index is empty."
                ],
            }

        doc_multivectors = self._document_multivectors(texts, batch_size=batch_size)

        doc_ids: List[str] = []
        fdes: List[np.ndarray] = []
        token_counts: List[int] = []

        for row, multivector in zip(valid_rows, doc_multivectors):
            tensor = self._to_2d_tensor(multivector)
            if tensor.numel() == 0:
                continue

            doc_id = row["id"]
            self._save_doc_multivector(doc_id, tensor)
            doc_ids.append(doc_id)
            token_counts.append(int(tensor.shape[0]))
            fdes.append(
                self._encoder().encode_document_multivectors(
                    tensor.detach().cpu().numpy().astype(np.float32)
                )
            )

        if not doc_ids:
            self.muvera_store.save_index(
                [],
                np.zeros((0, self._encoder().output_dim()), dtype=np.float32),
            )
            return {
                "status": "indexed",
                "indexed_docs": 0,
                "vector_dir": str(self.vector_dir),
                "muvera_vector_dim": self._encoder().output_dim(),
                "notes": [
                    "ColBERT document embedding produced no saved multivectors."
                ],
            }

        vectors = np.stack(fdes).astype(np.float32)
        self.muvera_store.save_index(doc_ids, vectors)

        return {
            "status": "indexed",
            "indexed_docs": len(doc_ids),
            "vector_dir": str(self.vector_dir),
            "muvera_vector_dim": int(vectors.shape[1]),
            "avg_tokens_per_doc": float(sum(token_counts) / len(token_counts)),
            "checkpoint": self.checkpoint_name,
            "notes": [
                "This index uses real ColBERT document multivectors saved to disk, then compresses them with MUVERA FDE.",
                "Search candidates can be reranked with real ColBERT MaxSim using the saved .pt files.",
            ],
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_k: int = 10,
        comparison_top_k: int | None = None,
    ) -> Dict[str, Any]:
        comparison_top_k = comparison_top_k or top_k
        baseline = self.search_service.search(query=query, top_k=comparison_top_k)
        expansion_phrases = self._query_expansion_phrases(query, baseline)
        query_variants = self._query_variants(query, expansion_phrases)

        if not query_variants:
            return {
                "query": query,
                "mode": "experimental_real_colbert_muvera",
                "counts": {"muvera_candidates": 0, "reranked": 0},
                "muvera_candidates": [],
                "reranked": [],
                "proxy_muvera": [],
                "dense": baseline.get("dense", []),
                "hybrid": baseline.get("hybrid", []),
                "notes": ["The query did not produce any ColBERT query vectors."],
            }

        primary_query_multivector = query_variants[0][1]
        candidate_pool_limit = max(top_k * 4, rerank_k * 2, 20)
        hits = self._retrieve_candidates(query_variants, top_k=candidate_pool_limit)
        candidates = self._join_hits_with_metadata(hits)[:candidate_pool_limit]
        reranked = self._rerank_candidates(
            query=query,
            expanded_phrases=expansion_phrases,
            query_variants=query_variants,
            candidates=candidates[:rerank_k],
        )

        notes = [
            "This path uses real ColBERT document/query multivectors, corpus-guided query expansion for candidate recall, and a composite rerank score.",
            "Compare this endpoint against /experimental/muvera/search to see the difference between proxy dense-span multivectors and real ColBERT multivectors.",
        ]
        proxy_hits: List[Dict[str, Any]] = []
        try:
            proxy = self.proxy_muvera_service.search(query=query, top_k=top_k)
            proxy_hits = proxy.get("muvera", [])
        except FileNotFoundError:
            notes.append(
                "Proxy MUVERA comparison is unavailable because the proxy index has not been built yet. Run POST /experimental/muvera/reindex if you want that comparison."
            )

        dense = baseline.get("dense", [])
        hybrid = baseline.get("hybrid", [])

        reranked_ids = {row["id"] for row in reranked}
        proxy_ids = {row["id"] for row in proxy_hits}
        dense_ids = {row["id"] for row in dense}
        hybrid_ids = {row["id"] for row in hybrid}

        return {
            "query": query,
            "mode": "experimental_real_colbert_muvera",
            "counts": {
                "muvera_candidates": len(candidates),
                "reranked": len(reranked),
                "proxy_muvera": len(proxy_hits),
                "dense": len(dense),
                "hybrid": len(hybrid),
                "overlap_proxy": len(reranked_ids & proxy_ids),
                "overlap_dense": len(reranked_ids & dense_ids),
                "overlap_hybrid": len(reranked_ids & hybrid_ids),
            },
            "colbert_muvera_config": {
                "checkpoint": self.checkpoint_name,
                "muvera_vector_dim": self._encoder().output_dim(),
                "query_tokens": int(primary_query_multivector.shape[0]),
                "vector_dir": str(self.vector_dir),
                "expansion_phrases": expansion_phrases,
                "query_variants": [variant for variant, _ in query_variants],
                "candidate_pool_limit": candidate_pool_limit,
            },
            "muvera_candidates": candidates,
            "reranked": reranked,
            "proxy_muvera": proxy_hits,
            "dense": dense,
            "hybrid": hybrid,
            "notes": notes,
        }

    def _checkpoint_instance(self) -> Checkpoint:
        if self._checkpoint is None:
            ensure_colbert_runtime_compatible()
            self._checkpoint = Checkpoint(
                self.checkpoint_name,
                colbert_config=ColBERTConfig(),
            )
        return self._checkpoint

    def _encoder(self) -> MuveraEncoder:
        if self._muvera_encoder is None:
            self._muvera_encoder = MuveraEncoder(dimension=128)
        return self._muvera_encoder

    def _query_multivector(self, query: str) -> torch.Tensor:
        tensor = self._checkpoint_instance().queryFromText([query], to_cpu=True)
        return self._to_2d_tensor(tensor)

    def _query_variants(
        self,
        query: str,
        expansion_phrases: List[str] | None = None,
    ) -> List[Tuple[str, torch.Tensor]]:
        variants = []
        for variant in self._expanded_queries(query, expansion_phrases or []):
            tensor = self._query_multivector(variant)
            if tensor.numel() == 0:
                continue
            variants.append((variant, tensor))
        return variants

    def _expanded_queries(
        self,
        query: str,
        expansion_phrases: List[str] | None = None,
    ) -> List[str]:
        expansion_phrases = expansion_phrases or []
        normalized = re.sub(r"\s+", " ", (query or "")).strip()
        if not normalized:
            return []

        variants: List[str] = []

        def add_variant(value: str) -> None:
            value = re.sub(r"\s+", " ", value).strip()
            if value and value not in variants:
                variants.append(value)

        add_variant(normalized)

        lowered = normalized.lower().rstrip("?.! ")
        stripped = re.sub(
            r"^(what is|what are|define|definition of|explain)\s+",
            "",
            lowered,
        ).strip()
        stripped = re.sub(r"^the\s+", "", stripped).strip()
        if stripped and stripped != lowered:
            add_variant(stripped)

        for phrase in expansion_phrases:
            add_variant(phrase)
            add_variant(f"{stripped or lowered} {phrase}")
            add_variant(f"{phrase} {stripped or lowered}")

        return variants[:12]

    def _query_expansion_phrases(
        self,
        query: str,
        baseline: Dict[str, Any],
    ) -> List[str]:
        rows = []
        seen_ids = set()
        for bucket in ("dense", "hybrid", "reranked"):
            for row in baseline.get(bucket, []):
                row_id = row.get("id")
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                rows.append(row)

        query_terms = {self._normalize_term(term) for term in content_terms(query)}
        candidates: Dict[str, float] = {}

        for rank, row in enumerate(rows[:6], start=1):
            heading = row.get("section_heading", "") or ""
            chunk_text = row.get("chunk_text", "") or ""
            sources = [
                (heading, 0.35),
                *[(sentence, 0.0) for sentence in self._split_sentences(chunk_text)[:4]],
            ]

            for text, source_bonus in sources:
                if source_bonus == 0.0 and lexical_overlap_count(query, text, heading="") == 0:
                    continue

                for phrase in self._candidate_phrases(text):
                    phrase_terms = [self._normalize_term(term) for term in phrase.split()]
                    if not phrase_terms:
                        continue
                    if all(term in query_terms for term in phrase_terms):
                        continue

                    novelty = sum(1 for term in phrase_terms if term not in query_terms)
                    if novelty == 0:
                        continue

                    overlap = sum(1 for term in phrase_terms if term in query_terms)
                    score = (
                        source_bonus
                        + (0.35 / rank)
                        + (0.18 * novelty)
                        + (0.08 * min(len(phrase_terms), 3))
                        + (0.10 * overlap)
                    )
                    if score > candidates.get(phrase, 0.0):
                        candidates[phrase] = score

        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        return [phrase for phrase, _ in ranked[:6]]

    def _candidate_phrases(self, text: str) -> List[str]:
        raw_tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        tokens = [
            token
            for token in raw_tokens
            if token
            and not token.isdigit()
            and token
            not in {
                "the",
                "a",
                "an",
                "and",
                "or",
                "of",
                "to",
                "for",
                "in",
                "on",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
            }
        ]
        phrases: List[str] = []

        for size in (3, 2):
            for idx in range(0, max(0, len(tokens) - size + 1)):
                phrase = " ".join(tokens[idx : idx + size]).strip()
                if phrase and phrase not in phrases:
                    phrases.append(phrase)

        return phrases[:8]

    def _split_sentences(self, text: str) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return []
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]

    def _retrieve_candidates(
        self,
        query_variants: List[Tuple[str, torch.Tensor]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        merged_hits: Dict[str, Dict[str, Any]] = {}

        for index, (variant, query_multivector) in enumerate(query_variants):
            query_fde = self._encoder().encode_query_multivectors(
                query_multivector.detach().cpu().numpy().astype(np.float32)
            )
            hits = self.muvera_store.search(query_fde, top_k=top_k)
            variant_weight = 1.0 if index == 0 else 0.97

            for hit in hits:
                adjusted_score = float(hit["muvera_score"]) * variant_weight
                existing = merged_hits.get(hit["id"])
                if existing and existing["muvera_score"] >= adjusted_score:
                    matched_queries = existing.setdefault("matched_queries", [])
                    if variant not in matched_queries:
                        matched_queries.append(variant)
                    continue

                merged_hits[hit["id"]] = {
                    **hit,
                    "muvera_score": adjusted_score,
                    "matched_query": variant,
                    "matched_queries": list(
                        {
                            *(existing.get("matched_queries", []) if existing else []),
                            variant,
                        }
                    ),
                }

        return sorted(
            merged_hits.values(),
            key=lambda item: item["muvera_score"],
            reverse=True,
        )

    def _document_multivectors(self, texts: List[str], batch_size: int) -> List[torch.Tensor]:
        raw = self._checkpoint_instance().docFromText(
            texts,
            bsize=batch_size,
            keep_dims=False,
            to_cpu=False,
            showprogress=False,
        )

        if isinstance(raw, tuple):
            raw = raw[0]

        if isinstance(raw, torch.Tensor):
            items = [raw[idx] for idx in range(raw.shape[0])]
        else:
            items = list(raw)

        return [self._to_2d_tensor(item) for item in items]

    def _to_2d_tensor(self, tensor) -> torch.Tensor:
        if isinstance(tensor, list):
            if not tensor:
                return torch.zeros((0, 128), dtype=torch.float32)
            tensor = torch.stack(
                [
                    item if isinstance(item, torch.Tensor) else torch.tensor(item)
                    for item in tensor
                ]
            )
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        elif not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)

        tensor = tensor.detach().cpu().float()
        if tensor.dim() == 3:
            tensor = tensor[0]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 0:
            return torch.zeros((0, 128), dtype=torch.float32)
        return tensor

    def _save_doc_multivector(self, doc_id: str, tensor: torch.Tensor) -> None:
        torch.save(tensor, self._vector_path(doc_id))

    def _load_doc_multivector(self, doc_id: str) -> torch.Tensor:
        return torch.load(self._vector_path(doc_id), map_location="cpu")

    def _vector_path(self, doc_id: str) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", doc_id).strip("_") or "doc"
        digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:8]
        return self.vector_dir / f"{safe_name}--{digest}.pt"

    def _join_hits_with_metadata(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        row_map = {
            row["id"]: row for row in filter_retrievable_rows(self.store.all_text_rows())
        }
        joined = []

        for hit in hits:
            row = row_map.get(hit["id"])
            if not row:
                continue

            joined.append(
                {
                    key: value
                    for key, value in {
                        **row,
                        **hit,
                        "colbert_vector_path": str(self._vector_path(hit["id"])),
                    }.items()
                    if key not in {"vector", "visual_vector"}
                }
            )

        return joined

    def _rerank_candidates(
        self,
        query: str,
        expanded_phrases: List[str],
        query_variants: List[Tuple[str, torch.Tensor]],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        scored = []
        for row in candidates:
            doc_multivector = self._load_doc_multivector(row["id"])
            variant_scores = {
                variant: self._maxsim_score(query_multivector, doc_multivector)
                for variant, query_multivector in query_variants
            }
            original_query = query_variants[0][0]
            colbert_score = variant_scores[original_query]
            best_variant, best_variant_score = max(
                variant_scores.items(),
                key=lambda item: item[1],
            )
            lexical_overlap = lexical_overlap_count(
                query=query,
                text=row.get("chunk_text", ""),
                heading=row.get("section_heading", ""),
            )
            query_coverage = self._query_coverage(query, row)
            answerability = self._answerability_score(
                query=query,
                row=row,
                expanded_phrases=expanded_phrases,
            )
            reference_penalty = self._reference_penalty(row)
            scored.append(
                {
                    **row,
                    "colbert_maxsim_score": colbert_score,
                    "best_variant_maxsim_score": best_variant_score,
                    "best_query_variant": best_variant,
                    "lexical_overlap": lexical_overlap,
                    "query_term_coverage": query_coverage,
                    "answerability_score": answerability,
                    "reference_penalty": reference_penalty,
                }
            )

        max_best_variant = max(
            (item["best_variant_maxsim_score"] for item in scored),
            default=1.0,
        ) or 1.0

        for item in scored:
            normalized_maxsim = item["best_variant_maxsim_score"] / max_best_variant
            composite_score = (
                (1.0 * normalized_maxsim)
                + (0.75 * item["query_term_coverage"])
                + (0.20 * min(item["lexical_overlap"], 4))
                + (0.90 * item["answerability_score"])
                - item["reference_penalty"]
            )
            item["composite_score"] = float(composite_score)

        scored.sort(key=lambda item: item["composite_score"], reverse=True)
        return scored

    def _query_coverage(self, query: str, row: Dict[str, Any]) -> float:
        query_terms = {self._normalize_term(term) for term in content_terms(query)}
        if not query_terms:
            return 0.0

        text_terms = {
            self._normalize_term(term)
            for term in content_terms(
                f"{row.get('section_heading', '')} {row.get('chunk_text', '')}"
            )
        }
        if not text_terms:
            return 0.0

        return len(query_terms & text_terms) / len(query_terms)

    def _answerability_score(
        self,
        query: str,
        row: Dict[str, Any],
        expanded_phrases: List[str],
    ) -> float:
        heading = (row.get("section_heading", "") or "").lower()
        text = (row.get("chunk_text", "") or "").lower()
        combined = f"{heading} {text}".strip()
        score = 0.0

        if self._is_definition_query(query) and re.search(
            r"\b(is|are|refers to|means|defined as|used to|uses|use|allows|provides|involves)\b",
            text,
        ):
            score += 0.25

        if len(self._split_sentences(text)) > 1:
            score += 0.10

        matched_expansions = sum(
            1 for phrase in expanded_phrases if phrase and phrase in combined
        )
        if matched_expansions:
            score += min(0.90, 0.22 * matched_expansions)

        return float(score)

    def _reference_penalty(self, row: Dict[str, Any]) -> float:
        heading = (row.get("section_heading", "") or "").lower()
        text = (row.get("chunk_text", "") or "").lower()
        penalty = 0.0

        if re.search(r"\b(reference|references|bibliography|works cited|further reading)\b", heading):
            penalty += 0.90

        citation_markers = len(re.findall(r"\[\d+\]|\(\d{4}\)|doi:|https?://|www\.", text))
        years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
        semicolons = text.count(";")
        if citation_markers + years >= 3:
            penalty += 0.35
        if semicolons >= 3:
            penalty += 0.15

        return min(1.25, penalty)

    def _is_definition_query(self, query: str) -> bool:
        normalized = (query or "").strip().lower()
        return normalized.startswith(DEFINITIONAL_PREFIXES)

    def _normalize_term(self, term: str) -> str:
        term = (term or "").strip().lower()
        if len(term) > 4 and term.endswith("s"):
            return term[:-1]
        return term

    def _maxsim_score(self, query_multivector: torch.Tensor, doc_multivector: torch.Tensor) -> float:
        query = self._to_2d_tensor(query_multivector)
        doc = self._to_2d_tensor(doc_multivector)
        if query.numel() == 0 or doc.numel() == 0:
            return 0.0

        similarity = torch.matmul(query, doc.transpose(0, 1))
        score = similarity.max(dim=1).values.sum().item()
        return float(score)

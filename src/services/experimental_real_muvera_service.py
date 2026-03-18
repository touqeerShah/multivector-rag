from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

from src.retrieval.colbert_service import ensure_colbert_runtime_compatible
from src.retrieval.muvera_encoder import MuveraEncoder
from src.retrieval.quality import filter_retrievable_rows
from src.retrieval.muvera_store import MuveraStore
from src.retrieval.store import RetrievalStore


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
            self.muvera_store.save_index([], np.zeros((0, self._encoder().output_dim()), dtype=np.float32))
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
            self.muvera_store.save_index([], np.zeros((0, self._encoder().output_dim()), dtype=np.float32))
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
        query_multivector = self._query_multivector(query)
        if query_multivector.numel() == 0:
            return {
                "query": query,
                "mode": "experimental_real_colbert_muvera",
                "counts": {"muvera_candidates": 0, "reranked": 0},
                "muvera_candidates": [],
                "reranked": [],
                "proxy_muvera": [],
                "dense": [],
                "hybrid": [],
                "notes": ["The query did not produce any ColBERT query vectors."],
            }

        query_fde = self._encoder().encode_query_multivectors(
            query_multivector.detach().cpu().numpy().astype(np.float32)
        )
        hits = self.muvera_store.search(query_fde, top_k=top_k)
        candidates = self._join_hits_with_metadata(hits)
        reranked = self._rerank_candidates(query_multivector, candidates[:rerank_k])

        baseline = self.search_service.search(query=query, top_k=comparison_top_k)
        notes = [
            "This path uses real ColBERT document/query multivectors and reranks MUVERA candidates with MaxSim.",
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
                "query_tokens": int(query_multivector.shape[0]),
                "vector_dir": str(self.vector_dir),
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
        row_map = {row["id"]: row for row in filter_retrievable_rows(self.store.all_text_rows())}
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
        query_multivector: torch.Tensor,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        scored = []
        for row in candidates:
            doc_multivector = self._load_doc_multivector(row["id"])
            score = self._maxsim_score(query_multivector, doc_multivector)
            scored.append(
                {
                    **row,
                    "colbert_maxsim_score": score,
                }
            )

        scored.sort(key=lambda item: item["colbert_maxsim_score"], reverse=True)
        return scored

    def _maxsim_score(self, query_multivector: torch.Tensor, doc_multivector: torch.Tensor) -> float:
        query = self._to_2d_tensor(query_multivector)
        doc = self._to_2d_tensor(doc_multivector)
        if query.numel() == 0 or doc.numel() == 0:
            return 0.0

        similarity = torch.matmul(query, doc.transpose(0, 1))
        score = similarity.max(dim=1).values.sum().item()
        return float(score)

from __future__ import annotations

import numpy as np

from src.services.experimental_muvera_service import ExperimentalMuveraService


class StubEmbedder:
    def embed_texts(self, texts):
        return [[float(len(text)), 1.0, 0.5] for text in texts]


class StubSearchService:
    def search(self, query: str, top_k: int = 10):
        return {
            "query": query,
            "dense": [{"id": "doc-1", "chunk_text": "alpha", "_score": 0.9}],
            "hybrid": [{"id": "doc-2", "chunk_text": "beta", "rrf_score": 0.8}],
            "reranked": [{"id": "doc-2", "chunk_text": "beta", "colbert_score": 1.2}],
        }


class StubStore:
    def all_text_rows(self):
        return [
            {
                "id": "doc-1",
                "doc_id": "a",
                "chunk_text": "alpha beta gamma",
                "vector": [0.1, 0.2, 0.3],
            },
            {
                "id": "doc-2",
                "doc_id": "b",
                "chunk_text": "delta epsilon zeta",
                "vector": [0.4, 0.5, 0.6],
            },
        ]


class StubMuveraEncoder:
    def encode_document_multivectors(self, multi_vectors):
        return np.array([multi_vectors.shape[0], 2.0], dtype=np.float32)

    def encode_query_multivectors(self, multi_vectors):
        return np.array([multi_vectors.shape[0], 2.0], dtype=np.float32)

    def output_dim(self):
        return 2


class StubMuveraStore:
    def __init__(self):
        self.saved_ids = None
        self.saved_vectors = None

    def save_index(self, doc_ids, vectors):
        self.saved_ids = doc_ids
        self.saved_vectors = vectors

    def search(self, query_fde, top_k=50):
        return [
            {"id": "doc-1", "rank": 1, "muvera_score": 0.99},
            {"id": "doc-2", "rank": 2, "muvera_score": 0.77},
        ][:top_k]


def test_muvera_reindex_builds_proxy_index():
    muvera_store = StubMuveraStore()
    service = ExperimentalMuveraService(
        embedder=StubEmbedder(),
        search_service=StubSearchService(),
        store=StubStore(),
        muvera_encoder=StubMuveraEncoder(),
        muvera_store=muvera_store,
    )

    result = service.rebuild_index(max_subvectors_per_doc=4)

    assert result["status"] == "indexed"
    assert result["indexed_docs"] == 2
    assert result["proxy_vector_dim"] == 2
    assert muvera_store.saved_ids == ["doc-1", "doc-2"]


def test_muvera_search_returns_comparison_without_vectors():
    service = ExperimentalMuveraService(
        embedder=StubEmbedder(),
        search_service=StubSearchService(),
        store=StubStore(),
        muvera_encoder=StubMuveraEncoder(),
        muvera_store=StubMuveraStore(),
    )

    result = service.search(query="alpha question", top_k=2)

    assert result["mode"] == "experimental_muvera_proxy"
    assert len(result["muvera"]) == 2
    assert "dense" in result
    assert "hybrid" in result
    assert "vector" not in result["muvera"][0]
    assert result["counts"]["overlap_dense"] == 1

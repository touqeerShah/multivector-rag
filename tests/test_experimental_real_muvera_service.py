from __future__ import annotations

import numpy as np
import torch

from src.services.experimental_real_muvera_service import ExperimentalRealMuveraService


class StubCheckpoint:
    def docFromText(self, docs, bsize=8, keep_dims=False, to_cpu=True, showprogress=False):
        return [
            torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0], [0.2, 0.8]], dtype=torch.float32),
        ][: len(docs)]

    def queryFromText(self, queries, to_cpu=True):
        return torch.tensor([[[1.0, 0.0], [0.5, 0.5]]], dtype=torch.float32)


class StubProxyMuveraService:
    def search(self, query: str, top_k: int = 10):
        return {
            "muvera": [
                {"id": "doc-2", "muvera_score": 0.8},
                {"id": "doc-1", "muvera_score": 0.7},
            ]
        }


class StubSearchService:
    def search(self, query: str, top_k: int = 10):
        return {
            "dense": [
                {
                    "id": "doc-1",
                    "_score": 0.9,
                    "section_heading": "Overview",
                    "chunk_text": "Alpha beta gamma delta explains the notice requirement.",
                }
            ],
            "hybrid": [
                {
                    "id": "doc-2",
                    "rrf_score": 0.8,
                    "section_heading": "Renewal",
                    "chunk_text": "Gamma delta epsilon zeta describes the renewal term.",
                }
            ],
        }


class MissingProxyMuveraService:
    def search(self, query: str, top_k: int = 10):
        raise FileNotFoundError("proxy MUVERA index missing")


class StubStore:
    def all_text_rows(self):
        return [
            {
                "id": "doc-1",
                "doc_id": "a",
                "chunk_text": "Alpha beta gamma delta explains the notice requirement.",
                "source_file": "docs/a.pdf",
                "page_number": 1,
                "vector": [0.1, 0.2],
            },
            {
                "id": "doc-2",
                "doc_id": "b",
                "chunk_text": "Gamma delta epsilon zeta describes the renewal term.",
                "source_file": "docs/b.pdf",
                "page_number": 2,
                "vector": [0.3, 0.4],
            },
        ]


class StubMuveraEncoder:
    def encode_document_multivectors(self, multi_vectors):
        return np.array([multi_vectors.shape[0], float(np.sum(multi_vectors))], dtype=np.float32)

    def encode_query_multivectors(self, multi_vectors):
        return np.array([multi_vectors.shape[0], float(np.sum(multi_vectors))], dtype=np.float32)

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
            {"id": "doc-2", "rank": 2, "muvera_score": 0.88},
        ][:top_k]


class StubCheckpointListOutput:
    def docFromText(self, docs, bsize=8, keep_dims=False, to_cpu=True, showprogress=False):
        assert to_cpu is False
        return [
            [
                torch.tensor([1.0, 0.0], dtype=torch.float32),
                torch.tensor([0.5, 0.5], dtype=torch.float32),
            ],
            [
                torch.tensor([0.0, 1.0], dtype=torch.float32),
                torch.tensor([0.2, 0.8], dtype=torch.float32),
            ],
        ][: len(docs)]

    def queryFromText(self, queries, to_cpu=True):
        return torch.tensor([[[1.0, 0.0], [0.5, 0.5]]], dtype=torch.float32)


class RankingStore:
    def all_text_rows(self):
        return [
            {
                "id": "references",
                "doc_id": "doc-ref",
                "section_heading": "References",
                "chunk_text": "Blockchain defined in prior work and cited references for blockchain systems.",
                "source_file": "docs/ref.pdf",
                "page_number": 30,
                "vector": [0.1, 0.1],
            },
            {
                "id": "blockchain",
                "doc_id": "doc-main",
                "section_heading": "Blockchain",
                "chunk_text": "Blockchain is a distributed ledger used to record transactions.",
                "source_file": "docs/main.pdf",
                "page_number": 13,
                "vector": [0.2, 0.2],
            },
            {
                "id": "cryptography",
                "doc_id": "doc-main",
                "section_heading": "2.1.2. Cryptography",
                "chunk_text": "Digital signatures use a private key and a public key to authenticate authenticity in blockchain transactions.",
                "source_file": "docs/main.pdf",
                "page_number": 11,
                "vector": [0.3, 0.3],
            },
        ]


class RankingCheckpoint:
    def docFromText(self, docs, bsize=8, keep_dims=False, to_cpu=True, showprogress=False):
        return [
            torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
            for _ in docs
        ]

    def queryFromText(self, queries, to_cpu=True):
        query = queries[0].lower()
        if "private key" in query or "digital signature" in query:
            return torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
        return torch.tensor([[[1.0, 0.0], [0.5, 0.5]]], dtype=torch.float32)


class RankingMuveraStore:
    def save_index(self, doc_ids, vectors):
        self.doc_ids = doc_ids
        self.vectors = vectors

    def search(self, query_fde, top_k=50):
        return [
            {"id": "references", "rank": 1, "muvera_score": 0.99},
            {"id": "blockchain", "rank": 2, "muvera_score": 0.97},
            {"id": "cryptography", "rank": 3, "muvera_score": 0.92},
        ][:top_k]


class AdaptiveSearchService:
    def search(self, query: str, top_k: int = 10):
        return {
            "dense": [
                {
                    "id": "cryptography",
                    "_score": 0.99,
                    "section_heading": "2.1.2. Cryptography",
                    "chunk_text": "Digital signatures use a private key and a public key to authenticate authenticity in blockchain transactions.",
                }
            ],
            "hybrid": [
                {
                    "id": "blockchain",
                    "rrf_score": 0.92,
                    "section_heading": "Blockchain",
                    "chunk_text": "Blockchain is a distributed ledger used to record transactions.",
                }
            ],
        }


def test_real_muvera_reindex_saves_colbert_multivectors(tmp_path):
    muvera_store = StubMuveraStore()
    service = ExperimentalRealMuveraService(
        search_service=StubSearchService(),
        proxy_muvera_service=StubProxyMuveraService(),
        store=StubStore(),
        vector_dir=str(tmp_path / "colbert_vectors"),
        checkpoint=StubCheckpoint(),
        muvera_encoder=StubMuveraEncoder(),
        muvera_store=muvera_store,
    )

    result = service.rebuild_index(batch_size=2)

    assert result["status"] == "indexed"
    assert result["indexed_docs"] == 2
    assert result["muvera_vector_dim"] == 2
    assert muvera_store.saved_ids == ["doc-1", "doc-2"]
    assert len(list((tmp_path / "colbert_vectors").glob("*.pt"))) == 2


def test_real_muvera_search_returns_reranked_real_colbert_results(tmp_path):
    muvera_store = StubMuveraStore()
    service = ExperimentalRealMuveraService(
        search_service=StubSearchService(),
        proxy_muvera_service=StubProxyMuveraService(),
        store=StubStore(),
        vector_dir=str(tmp_path / "colbert_vectors"),
        checkpoint=StubCheckpoint(),
        muvera_encoder=StubMuveraEncoder(),
        muvera_store=muvera_store,
    )
    service.rebuild_index(batch_size=2)

    result = service.search(query="alpha question", top_k=2, rerank_k=2)

    assert result["mode"] == "experimental_real_colbert_muvera"
    assert len(result["muvera_candidates"]) == 2
    assert len(result["reranked"]) == 2
    assert "colbert_maxsim_score" in result["reranked"][0]
    assert "proxy_muvera" in result
    assert "dense" in result
    assert "hybrid" in result
    assert "vector" not in result["muvera_candidates"][0]


def test_real_muvera_reindex_handles_list_output_from_colbert(tmp_path):
    muvera_store = StubMuveraStore()
    service = ExperimentalRealMuveraService(
        search_service=StubSearchService(),
        proxy_muvera_service=StubProxyMuveraService(),
        store=StubStore(),
        vector_dir=str(tmp_path / "colbert_vectors"),
        checkpoint=StubCheckpointListOutput(),
        muvera_encoder=StubMuveraEncoder(),
        muvera_store=muvera_store,
    )

    result = service.rebuild_index(batch_size=2)

    assert result["status"] == "indexed"
    assert result["indexed_docs"] == 2
    assert muvera_store.saved_ids == ["doc-1", "doc-2"]


def test_real_muvera_search_still_works_without_proxy_index(tmp_path):
    muvera_store = StubMuveraStore()
    service = ExperimentalRealMuveraService(
        search_service=StubSearchService(),
        proxy_muvera_service=MissingProxyMuveraService(),
        store=StubStore(),
        vector_dir=str(tmp_path / "colbert_vectors"),
        checkpoint=StubCheckpoint(),
        muvera_encoder=StubMuveraEncoder(),
        muvera_store=muvera_store,
    )
    service.rebuild_index(batch_size=2)

    result = service.search(query="alpha question", top_k=2, rerank_k=2)

    assert len(result["reranked"]) == 2
    assert result["proxy_muvera"] == []
    assert any("proxy index has not been built yet" in note for note in result["notes"])


def test_real_muvera_expands_signature_queries():
    service = ExperimentalRealMuveraService(
        search_service=AdaptiveSearchService(),
        proxy_muvera_service=StubProxyMuveraService(),
        store=StubStore(),
        checkpoint=StubCheckpoint(),
        muvera_encoder=StubMuveraEncoder(),
        muvera_store=StubMuveraStore(),
    )

    baseline = service.search_service.search("what is the blockchain signature?")
    expansion_phrases = service._query_expansion_phrases(
        "what is the blockchain signature?",
        baseline,
    )
    variants = service._expanded_queries(
        "what is the blockchain signature?",
        expansion_phrases,
    )

    assert "digital signatures use" in expansion_phrases or "digital signatures" in " ".join(expansion_phrases)
    assert any("private key" in variant for variant in variants)


def test_real_muvera_composite_rerank_penalizes_references_and_boosts_answerability(tmp_path):
    service = ExperimentalRealMuveraService(
        search_service=AdaptiveSearchService(),
        proxy_muvera_service=StubProxyMuveraService(),
        store=RankingStore(),
        vector_dir=str(tmp_path / "colbert_vectors"),
        checkpoint=RankingCheckpoint(),
        muvera_encoder=StubMuveraEncoder(),
        muvera_store=RankingMuveraStore(),
    )
    service.rebuild_index(batch_size=3)

    result = service.search(
        query="what is the blockchain signature?",
        top_k=3,
        rerank_k=3,
    )

    assert result["reranked"][0]["id"] == "cryptography"
    assert result["reranked"][-1]["id"] == "references"
    assert result["reranked"][0]["answerability_score"] > result["reranked"][-1]["answerability_score"]
    assert result["reranked"][-1]["reference_penalty"] > 0

from src.services.search_service import SearchService


class StubEmbedder:
    def embed_query(self, query: str):
        return [0.1, 0.2]


class StubStore:
    def all_text_rows(self):
        return [
            {
                "id": "doc-1",
                "chunk_text": "termination notice is 30 days",
                "section_heading": "Termination",
                "vector": [0.1, 0.2],
            }
        ]

    def text_vector_search(self, query_vector, top_k=10):
        return [
            {
                "id": "doc-1",
                "chunk_text": "termination notice is 30 days",
                "section_heading": "Termination",
                "vector": [0.1, 0.2],
                "_score": 0.9,
            }
        ]


class StubBM25:
    def build(self, rows):
        self.rows = rows

    def search(self, query: str, top_k: int = 10):
        return [
            {
                "id": "doc-1",
                "chunk_text": "termination notice is 30 days",
                "section_heading": "Termination",
                "vector": [0.1, 0.2],
                "bm25_score": 2.0,
            }
        ]


class StubRerankService:
    def rerank_text_candidates(self, query, candidates, top_k=10):
        return [
            {
                **candidates[0],
                "colbert_score": 3.0,
            }
        ]


def test_search_service_strips_vectors_from_api_payload():
    service = SearchService(embedder=StubEmbedder())
    service.store = StubStore()
    service.bm25 = StubBM25()
    service.rerank_service = StubRerankService()

    result = service.search(query="termination notice", top_k=5)

    for bucket in ["bm25", "dense", "hybrid", "reranked"]:
        assert result[bucket]
        assert "vector" not in result[bucket][0]

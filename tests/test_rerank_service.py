from src.services.rerank_service import RerankService


class StubColbert:
    def __init__(self):
        self.docs = None

    def rerank(self, query, docs, top_k=10):
        self.docs = docs
        return docs[:top_k]


def test_rerank_service_gates_to_lexically_relevant_candidates():
    service = RerankService()
    service.colbert = StubColbert()

    candidates = [
        {
            "id": "irrelevant",
            "chunk_text": "renewal clause and payment schedule",
            "section_heading": "Renewal",
        },
        {
            "id": "relevant",
            "chunk_text": "termination notice is 30 days in writing",
            "section_heading": "Termination",
        },
    ]

    result = service.rerank_text_candidates(
        query="termination notice",
        candidates=candidates,
        top_k=5,
    )

    assert [doc["id"] for doc in service.colbert.docs] == ["relevant"]
    assert result[0]["lexical_overlap"] >= 2


def test_rerank_service_falls_back_when_no_candidate_overlaps():
    service = RerankService()
    service.colbert = StubColbert()

    candidates = [
        {
            "id": "fallback",
            "chunk_text": "pricing schedule for annual billing",
            "section_heading": "Pricing",
        }
    ]

    result = service.rerank_text_candidates(
        query="termination notice",
        candidates=candidates,
        top_k=5,
    )

    assert [doc["id"] for doc in service.colbert.docs] == ["fallback"]
    assert result[0]["lexical_overlap"] == 0

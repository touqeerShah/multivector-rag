from src.services.answer_service import AnswerService


class StubSearchService:
    def search(self, query: str, top_k: int = 10):
        return {
            "query": query,
            "reranked": [
                {
                    "id": "a1",
                    "doc_id": "doc-a",
                    "source_file": "docs/a.pdf",
                    "page_number": 2,
                    "section_heading": "Termination Clause",
                    "chunk_text": (
                        "The agreement may be terminated by either party with 30 days written notice. "
                        "Termination for cause is immediate after material breach."
                    ),
                    "colbert_score": 4.2,
                },
                {
                    "id": "b1",
                    "doc_id": "doc-b",
                    "source_file": "docs/b.pdf",
                    "page_number": 5,
                    "section_heading": "Renewal",
                    "chunk_text": "The contract renews automatically unless notice is given before the renewal date.",
                    "colbert_score": 3.1,
                },
            ],
            "hybrid": [],
        }


def test_answer_service_returns_prompt_answer_and_citations():
    service = AnswerService(StubSearchService())

    result = service.answer(query="What is the termination notice period?", top_k=5, evidence_k=2)

    assert result["query"] == "What is the termination notice period?"
    assert "[1]" in result["answer"]
    assert len(result["citations"]) == 2
    assert "Question: What is the termination notice period?" in result["prompt"]
    assert "[1] doc_id=doc-a page=2" in result["prompt"]


def test_answer_service_handles_missing_evidence():
    class EmptySearchService:
        def search(self, query: str, top_k: int = 10):
            return {"query": query, "reranked": [], "hybrid": []}

    service = AnswerService(EmptySearchService())
    result = service.answer(query="Unknown question")

    assert "could not find enough indexed evidence" in result["answer"].lower()
    assert result["citations"] == []

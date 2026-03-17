from unittest.mock import patch

from src.services.indexing import IndexingService


class DummyEmbedder:
    def embed_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]


class DummyStore:
    def __init__(self):
        self.rows = None

    def add_text_rows(self, rows):
        self.rows = rows

    def all_text_rows(self):
        return self.rows or []


class DummyBM25:
    def __init__(self):
        self.rows = None

    def build(self, rows):
        self.rows = rows


def test_index_file_uses_text_store_api():
    service = IndexingService(embedder=DummyEmbedder())
    service.store = DummyStore()
    service.bm25 = DummyBM25()

    fake_rows = [{"id": "row-1", "chunk_text": "hello"}]
    service._build_rows_from_text = lambda doc_id, source_file, text: fake_rows

    with patch("src.services.indexing.extract_txt_text", return_value="hello world"):
        result = service.index_file("sample.txt")

    assert result["status"] == "indexed"
    assert service.store.rows == fake_rows
    assert service.bm25.rows == fake_rows

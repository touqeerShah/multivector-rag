from pathlib import Path
import lancedb


class RetrievalStore:
    def __init__(self, uri: str = "data/lancedb"):
        Path(uri).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(uri)

    def get_or_create_text_table(self):
        schema = [
            {"name": "id", "type": "string"},
            {"name": "doc_id", "type": "string"},
            {"name": "page_number", "type": "int32"},
            {"name": "chunk_text", "type": "string"},
            {"name": "vector", "type": "list<float32>"},
        ]
        try:
            return self.db.open_table("text_chunks")
        except Exception:
            return self.db.create_table("text_chunks", data=[])

    def get_or_create_pages_table(self):
        try:
            return self.db.open_table("page_images")
        except Exception:
            return self.db.create_table("page_images", data=[])

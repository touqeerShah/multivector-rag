from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self):
        self.docs = []
        self.doc_ids = []
        self.index = None

    def build(self, rows: list[dict]):
        self.docs = [r["chunk_text"].split() for r in rows]
        self.doc_ids = [r["id"] for r in rows]
        self.index = BM25Okapi(self.docs)

    def search(self, query: str, top_k: int = 10):
        if self.index is None:
            return []
        scores = self.index.get_scores(query.split())
        ranked = sorted(
            zip(self.doc_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        return [{"id": doc_id, "score": float(score)} for doc_id, score in ranked]
class ColBERTReranker:
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.model_name = model_name
        self.ready = False

    def load(self):
        self.ready = True

    def embed_query(self, query: str):
        raise NotImplementedError

    def embed_document(self, text: str):
        raise NotImplementedError

    def maxsim_score(self, q_vectors, d_vectors) -> float:
        raise NotImplementedError

    def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        raise NotImplementedError
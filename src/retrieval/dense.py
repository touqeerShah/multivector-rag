from sentence_transformers import SentenceTransformer
from typing import List
from functools import lru_cache


@lru_cache
def get_sentence_transformer(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    return SentenceTransformer(model_name)


class DenseEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = get_sentence_transformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()

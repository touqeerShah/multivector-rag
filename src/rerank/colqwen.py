from __future__ import annotations

from typing import List, Dict, Any

import torch
from PIL import Image


class ColQwen2Service:
    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        device: str | None = None,
    ):
        # lazy import so ColBERT-only env can still boot
        try:
            from colpali_engine.models import ColQwen2, ColQwen2Processor
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "colpali_engine is not installed in this environment. "
                "Use the ColPali environment for visual retrieval."
            ) from e

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).eval()
        self.model.to(self.device)

        self.processor = ColQwen2Processor.from_pretrained(model_name)

    def embed_query(self, query: str) -> List[float]:
        with torch.no_grad():
            batch = self.processor.process_queries([query]).to(self.device)
            embeddings = self.model(**batch)

        return self._pool_embedding(embeddings[0])

    def embed_page_image(self, image_path: str) -> List[float]:
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            batch = self.processor.process_images([image]).to(self.device)
            embeddings = self.model(**batch)

        return self._pool_embedding(embeddings[0])

    def score_query_to_pages(
        self,
        query: str,
        pages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        query_vec = torch.tensor(self.embed_query(query), dtype=torch.float32)

        scored: List[Dict[str, Any]] = []
        for page in pages:
            page_vec = torch.tensor(page["visual_vector"], dtype=torch.float32)
            score = torch.dot(query_vec, page_vec).item()
            scored.append({**page, "visual_score": float(score)})

        scored.sort(key=lambda x: x["visual_score"], reverse=True)
        return scored

    def _pool_embedding(self, embedding_tensor) -> List[float]:
        if embedding_tensor.dim() == 2:
            pooled = embedding_tensor.mean(dim=0)
        else:
            pooled = embedding_tensor.squeeze(0).mean(dim=0)

        pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
        return pooled.detach().cpu().float().tolist()
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class DenseTextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            out = self.model(**encoded)
            cls = out.last_hidden_state[:, 0]
            cls = F.normalize(cls, p=2, dim=1)
        return cls.cpu().tolist()
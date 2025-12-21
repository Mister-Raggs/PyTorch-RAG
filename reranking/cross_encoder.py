import torch
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query, chunks, top_k=None):
        """
        query: str
        chunks: list of dicts (output of retrieve())
        """
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)

        for c, s in zip(chunks, scores):
            c["rerank_score"] = float(s)

        chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        if top_k:
            chunks = chunks[:top_k]

        return chunks

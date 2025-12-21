import json
import os
import numpy as np
import torch
from numpy.linalg import norm
from pathlib import Path

# Optional local dependencies
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from huggingface_hub import InferenceApi
except Exception:
    InferenceApi = None

from reranking.cross_encoder import CrossEncoderReranker

# -----------------------------
# Paths
# -----------------------------
# Get project root (parent of indexing/)
PROJECT_ROOT = Path(__file__).parent.parent
EMBED_DIR = PROJECT_ROOT / "data/processed/embeddings"
CHUNK_FILE = PROJECT_ROOT / "data/processed/chunks.json"

# -----------------------------
# Load data
# -----------------------------
with open(CHUNK_FILE, "r") as f:
    chunks = json.load(f)

doc_titles = {}
for c in chunks:
    if c.get("title"):
        doc_titles[c["doc_id"]] = c["title"]

embeddings = np.load(EMBED_DIR / "embeddings.npy")
chunk_ids = json.load(open(EMBED_DIR / "chunk_ids.json"))

# Map chunk_id → chunk
id2chunk = {c["chunk_id"]: c for c in chunks}

# Normalize embeddings for cosine similarity
embeddings = embeddings / norm(embeddings, axis=1, keepdims=True)

# -----------------------------
# Query embedding (local or remote)
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
USE_REMOTE_EMBED = os.environ.get("USE_REMOTE_EMBED", "0").lower() in {"1", "true", "yes"}
HF_TOKEN = os.environ.get("HF_HUB_TOKEN")

device = "mps" if torch.backends.mps.is_available() else "cpu"

def encode_query(query: str) -> np.ndarray:
    """Return a normalized embedding vector for the query.

    Uses Hugging Face Inference API when USE_REMOTE_EMBED=1 and token is provided; otherwise
    falls back to local SentenceTransformer.
    """
    if USE_REMOTE_EMBED:
        if InferenceApi is None:
            raise RuntimeError("huggingface-hub not installed; cannot use remote embedding. Install huggingface-hub or disable USE_REMOTE_EMBED.")
        if not HF_TOKEN:
            raise RuntimeError("HF_HUB_TOKEN not found; set it in environment or .env to use remote embedding.")
        client = InferenceApi(repo_id=MODEL_NAME, token=HF_TOKEN, task="feature-extraction")
        vec = client(inputs=query)
        v = np.array(vec, dtype=np.float32)
        if v.ndim == 2:
            v = v.mean(axis=0)
        v = v / norm(v)
        return v
    else:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed; install it or set USE_REMOTE_EMBED=1 to use remote embedding.")
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        q_vec = model.encode([query], convert_to_numpy=True)
        q_vec = q_vec / norm(q_vec)
        return q_vec.flatten()
# -----------------------------
# Retrieval
# -----------------------------
def resolve_title(chunk):
    return (
        chunk["metadata"].get("section")
        or doc_titles.get(chunk["doc_id"])
        or f"{chunk['source']}::{chunk['doc_id'][:8]}"
    )

def retrieve(query, top_k=5, rerank=False, rerank_k=10):
    q_vec = encode_query(query)

    sims = embeddings @ q_vec.T
    idxs = sims.flatten().argsort()[::-1][:top_k]
    results = []
    for i in idxs:
      chunk = id2chunk[chunk_ids[i]]
      results.append({
          "score": float(sims[i]),
          "text": chunk["text"],
          "title": resolve_title(chunk),
          "source": chunk["source"],
          "chunk_strategy": chunk["metadata"].get("chunk_strategy"),
          "token_count": chunk["metadata"].get("token_count"),
      })

    # Optional reranking
    if rerank:
        # Cross-encoder reranking requires local model download; keep optional.
        # If USE_REMOTE_EMBED is enabled but local reranker isn't available, skip.
        try:
            results = CrossEncoderReranker().rerank(query, results, top_k=top_k)
        except Exception:
            # Fallback: return as-is if reranker cannot be used
            pass

    return results

# -----------------------------
# Pretty printer (NEW)
# -----------------------------
def print_results(query, results):
    print(f"\n🔍 Query: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"#{i} | score={r['score']:.4f}")
        print(f"  strategy     : {r['chunk_strategy']}")
        print(f"  tokens       : {r['token_count']}")
        print(f"  source       : {r['source']}")
        print(f"  title        : {r['title']}")
        print(f"  text preview : {r['text']}")
        print("-" * 60)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    query = "How do I use autograd in PyTorch?"
    results = retrieve(query, top_k=6)
    print_results(query, results)

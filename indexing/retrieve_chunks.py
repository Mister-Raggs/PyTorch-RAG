import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

# -----------------------------
# Paths
# -----------------------------
EMBED_DIR = "data/processed/embeddings"
CHUNK_FILE = "data/processed/chunks.json"

# -----------------------------
# Load data
# -----------------------------
with open(CHUNK_FILE, "r") as f:
    chunks = json.load(f)

doc_titles = {}
for c in chunks:
    if c.get("title"):
        doc_titles[c["doc_id"]] = c["title"]

embeddings = np.load(f"{EMBED_DIR}/embeddings.npy")
chunk_ids = json.load(open(f"{EMBED_DIR}/chunk_ids.json"))

# Map chunk_id → chunk
id2chunk = {c["chunk_id"]: c for c in chunks}

# Normalize embeddings for cosine similarity
embeddings = embeddings / norm(embeddings, axis=1, keepdims=True)

# -----------------------------
# Load model (M1-safe)
# -----------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# -----------------------------
# Retrieval
# -----------------------------
def resolve_title(chunk):
    return (
        chunk["metadata"].get("section")
        or doc_titles.get(chunk["doc_id"])
        or f"{chunk['source']}::{chunk['doc_id'][:8]}"
    )

def retrieve(query, top_k=5):
    q_vec = model.encode([query], convert_to_numpy=True)
    q_vec = q_vec / norm(q_vec)

    sims = embeddings @ q_vec.T
    idxs = sims.flatten().argsort()[::-1][:top_k]

    results = []
    for i in idxs:
        chunk = id2chunk[chunk_ids[i]]
        results.append({
            "score": float(sims[i]),
            "text": chunk["text"][:300] + "...",
            "title": resolve_title(chunk),
            "source": chunk.get("source"),
            "chunk_strategy": chunk["metadata"].get("chunk_strategy"),
            "token_count": chunk["metadata"].get("token_count")
        })
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

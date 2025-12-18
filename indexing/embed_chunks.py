import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

CHUNK_FILE = "data/processed/chunks.json"
EMBED_DIR = "data/processed/embeddings"
BATCH_SIZE = 32
MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs(EMBED_DIR, exist_ok=True)

# Load chunks
with open(CHUNK_FILE) as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]
chunk_ids = [c["chunk_id"] for c in chunks]

print(f"[INFO] Loaded {len(texts)} chunks")

# Load MiniLM
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {device}")
model = SentenceTransformer(MODEL_NAME, device=device)

# Embedding loop
embeddings = []
for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_emb = model.encode(batch_texts, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=True)
    embeddings.append(batch_emb)

embeddings = np.vstack(embeddings).astype(np.float32)

# Save embeddings and IDs
np.save(os.path.join(EMBED_DIR, "embeddings.npy"), embeddings)
with open(os.path.join(EMBED_DIR, "chunk_ids.json"), "w") as f:
    json.dump(chunk_ids, f)

print(f"[INFO] Saved embeddings ({embeddings.shape}) and chunk_ids")

import json
from chunking import (
    fixed_chunking,
    fixed_overlap_chunking,
    header_chunking,
    hybrid_chunking
)

docs = json.load(open("data/processed/corpus.json"))

all_chunks = []

for doc in docs:
    all_chunks.extend(fixed_chunking(doc))
    all_chunks.extend(fixed_overlap_chunking(doc))
    all_chunks.extend(header_chunking(doc))
    all_chunks.extend(hybrid_chunking(doc))

with open("data/processed/chunks.json", "w") as f:
    json.dump(all_chunks, f, indent=2)

import json
import glob
import os

all_docs = []

for path in glob.glob("data/raw/docs/*.json"):
    with open(path) as f:
        all_docs.append(json.load(f))

for path in glob.glob("data/raw/issues/*.json"):
    with open(path) as f:
        all_docs.append(json.load(f))

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/corpus.json", "w") as f:
    json.dump(all_docs, f, indent=2)

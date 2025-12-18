import json
import glob


def load_all():
    corpus = []

    for path in glob.glob("data/raw/docs/*.json"):
        corpus.append(json.load(open(path)))

    for path in glob.glob("data/raw/issues/*.json"):
        corpus.append(json.load(open(path)))

    return corpus


def main():
    corpus = load_all()

    assert len(corpus) > 0, "Corpus is empty — raw data missing?"

    print(f"[INFO] Built corpus with {len(corpus)} documents")

    with open("data/processed/corpus.json", "w") as f:
        json.dump(corpus, f, indent=2)


if __name__ == "__main__":
    main()

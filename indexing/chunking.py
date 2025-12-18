import re
import uuid
from typing import List, Dict
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def make_chunk(doc, text, strategy, section=None):
    return {
        "chunk_id": str(uuid.uuid4()),
        "doc_id": doc["doc_id"],
        "source": doc["source"],
        "text": text.strip(),
        "metadata": {
            "chunk_strategy": strategy,
            "section": section,
            "token_count": count_tokens(text)
        }
    }

def fixed_chunking(doc, max_tokens=300) -> List[Dict]:
    tokens = enc.encode(doc["text"])
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_text = enc.decode(tokens[i:i + max_tokens])
        chunks.append(
            make_chunk(doc, chunk_text, "fixed")
        )

    return chunks

def fixed_overlap_chunking(doc, max_tokens=300, overlap=50) -> List[Dict]:
    tokens = enc.encode(doc["text"])
    chunks = []
    step = max_tokens - overlap

    for i in range(0, len(tokens), step):
        chunk_text = enc.decode(tokens[i:i + max_tokens])
        chunks.append(
            make_chunk(doc, chunk_text, "fixed_overlap")
        )

    return chunks

HEADER_RE = re.compile(r"\n##+\s+")


def header_chunking(doc, max_tokens=500) -> List[Dict]:
    sections = HEADER_RE.split(doc["text"])
    headers = HEADER_RE.findall(doc["text"])

    chunks = []

    for i, section_text in enumerate(sections):
        section_text = section_text.strip()
        if not section_text:
            continue

        header = headers[i - 1].strip() if i > 0 else None

        if count_tokens(section_text) <= max_tokens:
            chunks.append(
                make_chunk(doc, section_text, "header", section=header)
            )
        else:
            # fallback to fixed chunking
            sub_doc = {**doc, "text": section_text}
            chunks.extend(
                fixed_chunking(sub_doc, max_tokens=max_tokens)
            )

    return chunks

def hybrid_chunking(doc, max_tokens=400, overlap=50) -> List[Dict]:
    sections = HEADER_RE.split(doc["text"])
    headers = HEADER_RE.findall(doc["text"])

    chunks = []

    for i, section_text in enumerate(sections):
        section_text = section_text.strip()
        if not section_text:
            continue

        header = headers[i - 1].strip() if i > 0 else None
        tokens = enc.encode(section_text)

        if len(tokens) <= max_tokens:
            chunks.append(
                make_chunk(doc, section_text, "hybrid", section=header)
            )
        else:
            step = max_tokens - overlap
            for j in range(0, len(tokens), step):
                sub_text = enc.decode(tokens[j:j + max_tokens])
                chunks.append(
                    make_chunk(doc, sub_text, "hybrid", section=header)
                )

    return chunks

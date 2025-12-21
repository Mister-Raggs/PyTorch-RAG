# generation/prompt_templates.py

RAG_PROMPT = """You are a PyTorch expert assistant.

Answer the question using ONLY the provided context.
If the answer cannot be determined from the context, say:
"I don't have enough information in the provided sources."

Context:
{context}

Question:
{question}

Answer:
"""


def format_chunks(chunks):
    formatted = []
    for i, ch in enumerate(chunks, 1):
        header = f"[Source {i}]"
        meta = []

        if "section" in ch["metadata"]:
            meta.append(f"Section: {ch['metadata']['section']}")
        if "url" in ch:
            meta.append(f"URL: {ch['url']}")

        meta_str = " | ".join(meta)

        formatted.append(
            f"{header}\n{meta_str}\n{ch['text']}"
        )

    return "\n\n".join(formatted)

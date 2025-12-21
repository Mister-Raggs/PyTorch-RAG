- GitHub’s Issues API returns pull request discussions; naive filtering removed all PyTorch issues. We retained PR-linked discussions to preserve high-quality maintainer explanations while filtering empty threads.

- We enforced a canonical document contract at ingestion time to prevent downstream schema drift.

- We ablated chunking strategies under identical retrieval settings to isolate their impact on Recall@k.

- I developed locally on Apple Silicon and scaled embedding generation to a cloud GPU via VS Code Remote SSH, keeping the same codebase and pipeline.

- Before adding a reranker, we evaluated chunking strategies and observed that header and hybrid chunking dominate early ranks, while fixed-overlap improves recall but harms precision.

## Remote-First Usage (No Local Model Downloads)

- Generation: The `RAGGenerator` uses Hugging Face Inference API (serverless). It first tries `text_generation`; if the provider only exposes `conversational` (e.g., `HuggingFaceH4/zephyr-7b-beta`), it falls back to `chat_completion` automatically.
- Retrieval: Set `USE_REMOTE_EMBED=1` to embed queries remotely with `sentence-transformers/all-MiniLM-L6-v2` and compare against precomputed chunk embeddings.
- Reranking: Cross-encoder reranking requires local downloads; keep it disabled or switch to a hosted rerank provider (e.g., Cohere/Jina) if desired.

### Model selection tips
- For text-generation endpoint: use `mistralai/Mistral-7B-Instruct-v0.2` or `tiiuae/falcon-7b-instruct`.
- For chat-only endpoint: `HuggingFaceH4/zephyr-7b-beta` will report `Endpoint used: chat_completion` in the notebook output. This is expected.

### Quick Start
- Create a `.env` with `HF_HUB_TOKEN=<your_token>`.
- Ensure minimal deps: see `requirements.txt` and install.
- In notebook or environment, set `USE_REMOTE_EMBED=1` to avoid local embedding.
- In `test_rag_generation.ipynb`, set `model_name` to a text-generation-friendly model if you want to avoid chat fallback.

### Install (minimal)
```
pip install -r requirements.txt
```

### Notes
- For local embeddings/reranking, additionally install `sentence-transformers` and `torch`.
- Remote inference has rate limits and context-size constraints depending on the model/provider.
- Notebook prints `Model used` and `Endpoint used` so you can verify whether `text_generation` or `chat_completion` ran.
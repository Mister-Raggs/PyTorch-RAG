- GitHub’s Issues API returns pull request discussions; naive filtering removed all PyTorch issues. We retained PR-linked discussions to preserve high-quality maintainer explanations while filtering empty threads.

- We enforced a canonical document contract at ingestion time to prevent downstream schema drift.

- We ablated chunking strategies under identical retrieval settings to isolate their impact on Recall@k.

- I developed locally on Apple Silicon and scaled embedding generation to a cloud GPU via VS Code Remote SSH, keeping the same codebase and pipeline.
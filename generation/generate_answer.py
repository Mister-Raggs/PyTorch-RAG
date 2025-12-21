# generation/generate_answer.py

from huggingface_hub import InferenceClient
from generation.prompt_templates import RAG_PROMPT


def format_chunks(chunks):
    """
    Formats retrieved chunks into a readable context block.
    """
    formatted = []
    for i, ch in enumerate(chunks, 1):
        title = ch.get("title", "Unknown")
        source = ch.get("source", "N/A")
        strategy = ch.get("chunk_strategy", "N/A")

        formatted.append(
            f"[Source {i}]\n"
            f"Title: {title} | Source: {source} | Strategy: {strategy}\n"
            f"{ch['text']}"
        )
    return "\n\n".join(formatted)


class RAGGenerator:
    """
    RAG generator backed by Hugging Face Inference API.
    """

    def __init__(self, model_name: str, hf_token: str, fallback_model: str | None = "HuggingFaceH4/zephyr-7b-beta"):
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.token = hf_token
        self.client = InferenceClient(model=model_name, token=hf_token)

    def generate(
        self,
        query: str,
        chunks: list,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ):
        context = format_chunks(chunks)

        prompt = RAG_PROMPT.format(
            context=context,
            question=query
        )

        endpoint = "text_generation"

        try:
            raw_text = self.client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
            )
            answer = raw_text.strip()
        except ValueError as ve:
            msg = str(ve)
            if "not supported for task text-generation" in msg and "conversational" in msg:
                endpoint = "chat_completion"
                chat = self.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                answer = chat["choices"][0]["message"]["content"].strip()
            else:
                raise ve
        except StopIteration:
            if self.fallback_model and self.fallback_model != self.model_name:
                self.client = InferenceClient(model=self.fallback_model, token=self.token)
                self.model_name = self.fallback_model
                endpoint = "text_generation"
                try:
                    raw_text = self.client.text_generation(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0.0,
                    )
                    answer = raw_text.strip()
                except ValueError as ve:
                    msg = str(ve)
                    if "not supported for task text-generation" in msg and "conversational" in msg:
                        endpoint = "chat_completion"
                        chat = self.client.chat_completion(
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_new_tokens,
                            temperature=temperature,
                        )
                        answer = chat["choices"][0]["message"]["content"].strip()
                    else:
                        raise ve
            else:
                raise

        return {
            "prompt": prompt,
            "answer": answer,
            "chunks": chunks,
            "model": self.model_name,
            "endpoint": endpoint,
        }

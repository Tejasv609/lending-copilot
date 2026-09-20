"""LLM client for any OpenAI-compatible API. Falls back to extractive answers when unconfigured."""
from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PROMPT = """You are Lending Copilot, an assistant that answers questions about lending and credit policy.
Answer ONLY using the provided context. If the context does not contain the answer, say so explicitly.
Cite every factual claim with the source document title in square brackets, e.g. [Retail Credit Policy].
Keep answers concise and factual."""


@dataclass
class LLMAnswer:
    text: str
    model: str
    used_llm: bool


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float = 0.1, max_tokens: int = 512) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def chat(self, question: str, context_blocks: list[str]) -> LLMAnswer:
        if not self.configured:
            return self._extractive(question, context_blocks)
        try:
            from openai import OpenAI
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            context = "\n\n".join(context_blocks)
            resp = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            )
            return LLMAnswer(text=resp.choices[0].message.content or "", model=self.model, used_llm=True)
        except Exception as exc:  # never fail the request because the LLM is down
            fallback = self._extractive(question, context_blocks)
            fallback.text = (f"(LLM unavailable: {exc}. Showing extractive answer.)\n\n" + fallback.text)
            return fallback

    def _extractive(self, question: str, context_blocks: list[str]) -> LLMAnswer:
        # No LLM configured: return the retrieved source excerpts verbatim with citations.
        body = "\n\n".join(context_blocks)
        return LLMAnswer(
            text=f"Based on the retrieved policy documents (no LLM configured, showing source excerpts):\n\n{body}",
            model="extractive",
            used_llm=False,
        )

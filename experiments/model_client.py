from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    raw: dict | None = None


class BaseModelClient:
    def generate(self, prompt: str, model: str, temperature: float, max_tokens: int) -> GenerationResult:
        raise NotImplementedError


class MockModelClient(BaseModelClient):
    """Deterministic local stub to validate experiment flow without API cost."""

    def generate(self, prompt: str, model: str, temperature: float, max_tokens: int) -> GenerationResult:
        lower = prompt.lower()
        mentions_negative = any(token in lower for token in ["negative", "complaint", "issue", "bad", "poor"])
        if mentions_negative:
            summary = (
                "Most reviews are positive about quality and overall experience, "
                "but one reviewer reports a negative issue that should be considered."
            )
        else:
            summary = "Reviews are mostly positive overall, with generally favorable customer sentiment."
        return GenerationResult(text=summary)


class OpenAIChatClient(BaseModelClient):
    def __init__(self):
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install the openai package: pip install openai") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:  # pragma: no cover
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, model: str, temperature: float, max_tokens: int) -> GenerationResult:
        response = self._client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        text = getattr(response, "output_text", "") or ""
        return GenerationResult(text=text, raw={"id": getattr(response, "id", None)})


def build_model_client(provider: str) -> BaseModelClient:
    provider_norm = provider.strip().lower()
    if provider_norm == "mock":
        return MockModelClient()
    if provider_norm == "openai":
        return OpenAIChatClient()
    raise ValueError(f"Unsupported provider '{provider}'. Use 'mock' or 'openai'.")

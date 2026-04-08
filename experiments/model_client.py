from __future__ import annotations

import os
from dataclasses import dataclass


OPENROUTER_MODELS = {
    "gemini-2.0-flash-lite": "google/gemini-2.0-flash-lite-001",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
}

MODEL_ALIASES = {
    # User-friendly names
    "gemini 2.0 flash-lite": OPENROUTER_MODELS["gemini-2.0-flash-lite"],
    "gemini 2.0 flash": OPENROUTER_MODELS["gemini-2.0-flash"],
    "gpt-4o mini": OPENROUTER_MODELS["gpt-4o-mini"],
    "claude 3 haiku": OPENROUTER_MODELS["claude-3-haiku"],
    "claude 3.7 sonnet": OPENROUTER_MODELS["claude-3.7-sonnet"],
    # Kebab-case aliases
    "gemini-2.0-flash-lite": OPENROUTER_MODELS["gemini-2.0-flash-lite"],
    "gemini-2.0-flash": OPENROUTER_MODELS["gemini-2.0-flash"],
    "gpt-4o-mini": OPENROUTER_MODELS["gpt-4o-mini"],
    "claude-3-haiku": OPENROUTER_MODELS["claude-3-haiku"],
    "claude-3.7-sonnet": OPENROUTER_MODELS["claude-3.7-sonnet"],
}


@dataclass
class GenerationResult:
    text: str
    raw: dict | None = None


class BaseModelClient:
    def generate(self, prompt: str, model: str, temperature: float, max_tokens: int) -> GenerationResult:
        raise NotImplementedError

    @staticmethod
    def resolve_model(model: str) -> str:
        key = model.strip().lower()
        return MODEL_ALIASES.get(key, model)


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
            model=self.resolve_model(model),
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        text = getattr(response, "output_text", "") or ""
        return GenerationResult(text=text, raw={"id": getattr(response, "id", None)})


class OpenRouterChatClient(BaseModelClient):
    def __init__(self):
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install the openai package: pip install openai") from exc

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:  # pragma: no cover
            raise RuntimeError("OPENROUTER_API_KEY is not set.")

        site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
        app_name = os.getenv("OPENROUTER_APP_NAME", "nlp_project")

        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            },
        )

    def generate(self, prompt: str, model: str, temperature: float, max_tokens: int) -> GenerationResult:
        resolved_model = self.resolve_model(model)
        response = self._client.responses.create(
            model=resolved_model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        text = getattr(response, "output_text", "") or ""
        return GenerationResult(
            text=text,
            raw={"id": getattr(response, "id", None), "model": resolved_model},
        )


def build_model_client(provider: str) -> BaseModelClient:
    provider_norm = provider.strip().lower()
    if provider_norm == "mock":
        return MockModelClient()
    if provider_norm == "openrouter":
        return OpenRouterChatClient()
    if provider_norm == "openai":
        return OpenAIChatClient()
    raise ValueError(f"Unsupported provider '{provider}'. Use 'mock', 'openrouter', or 'openai'.")

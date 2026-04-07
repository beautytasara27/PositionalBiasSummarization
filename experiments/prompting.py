from __future__ import annotations


STRATEGIES = ("baseline", "cot", "repetition")


def build_prompt(strategy: str, concatenated_reviews: str) -> str:
    strategy = strategy.strip().lower()

    if strategy == "baseline":
        return (
            "Summarize the customer opinions in 3-5 concise sentences. "
            f"Reviews:\n{concatenated_reviews}"
        )

    if strategy == "cot":
        return (
            "Analyze the reviews step-by-step internally, then provide ONLY the final summary in 3-5 sentences. "
            f"Reviews:\n{concatenated_reviews}"
        )

    if strategy == "repetition":
        return (
            "Summarize in 3-5 sentences. Mention any negative/minority opinion if present. "
            f"Reviews:\n{concatenated_reviews}"
        )

    raise ValueError(f"Unsupported strategy '{strategy}'.")

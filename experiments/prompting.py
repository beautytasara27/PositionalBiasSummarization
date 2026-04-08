from __future__ import annotations


STRATEGIES = ("baseline", "cot", "repetition")


def build_prompt(strategy: str, concatenated_reviews: str) -> str:
    strategy = strategy.strip().lower()

    if strategy == "baseline":
        return (
            "Summarize the customer opinions in 2 concise sentences. "
            f"Reviews:\n{concatenated_reviews}"
        )

    if strategy == "cot":
        return (
            "Summarize the customer opinions in 2 concise sentences. Think step by step "
            f"Reviews:\n{concatenated_reviews}"
        )

    if strategy == "repetition":
        return (
            "Summarize the customer opinions in 2 concise sentences."
            f"Reviews:\n{concatenated_reviews}"
            "Summarize the customer opinions in 2 concise sentences."
            f"Reviews:\n{concatenated_reviews}"
        )

    raise ValueError(f"Unsupported strategy '{strategy}'.")

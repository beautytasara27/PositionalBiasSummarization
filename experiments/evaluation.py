from __future__ import annotations

import re
from collections import Counter


NEGATIVE_MARKERS = {
    "bad",
    "poor",
    "negative",
    "issue",
    "problem",
    "complaint",
    "slow",
    "rude",
    "disappoint",
    "worst",
    "unhappy",
    "however",
    "but",
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "is",
    "it",
    "this",
    "that",
    "with",
    "was",
    "are",
    "as",
    "at",
    "be",
    "very",
    "from",
    "they",
    "we",
    "i",
    "my",
    "our",
    "their",
}


def split_reviews(concatenated_text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", concatenated_text) if part.strip()]
    return parts


def get_negative_review_by_position(concatenated_text: str, dataset_position: str) -> str:
    reviews = split_reviews(concatenated_text)
    if not reviews:
        return ""

    if dataset_position == "top":
        return reviews[0]
    if dataset_position == "middle":
        return reviews[len(reviews) // 2]
    if dataset_position == "end":
        return reviews[-1]
    raise ValueError(f"Unknown dataset position '{dataset_position}'.")


def extract_keywords(text: str, max_terms: int = 8) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    filtered = [tok for tok in tokens if tok not in STOPWORDS]
    common = [token for token, _ in Counter(filtered).most_common(max_terms)]
    return common


def minority_recall(summary: str, negative_review_text: str) -> tuple[int, int, list[str]]:
    summary_l = summary.lower()
    cue_keywords = extract_keywords(negative_review_text)
    matched_keywords = [kw for kw in cue_keywords if kw in summary_l]
    has_keyword_match = int(bool(matched_keywords))
    has_negative_marker = int(any(marker in summary_l for marker in NEGATIVE_MARKERS))
    # Primary recall: summary must either reflect specific negative content or clearly acknowledge negative minority signal.
    primary_recall = int(bool(has_keyword_match or has_negative_marker))
    return primary_recall, has_keyword_match, matched_keywords


def output_length_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def compression_ratio(input_text: str, output_text: str) -> float:
    in_words = max(output_length_words(input_text), 1)
    out_words = output_length_words(output_text)
    return out_words / in_words

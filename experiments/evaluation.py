from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from statistics import mean


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
    "there",
    "here",
    "had",
    "has",
    "have",
    "been",
    "were",
    "will",
    "would",
    "could",
    "should",
    "do",
    "does",
    "did",
}


@lru_cache(maxsize=1)
def _load_vader_analyzer():
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install nltk to use VADER scoring.") from exc

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        import nltk

        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def split_reviews(concatenated_text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", concatenated_text) if part.strip()]


def score_text_vader(text: str) -> float:
    analyzer = _load_vader_analyzer()
    return float(analyzer.polarity_scores(text or "").get("compound", 0.0))


def score_reviews_vader(reviews: list[str]) -> list[float]:
    return [score_text_vader(review) for review in reviews]


def expected_aggregate_sentiment(review_scores: list[float]) -> float:
    return float(mean(review_scores)) if review_scores else 0.0


def sentiment_deviation(summary_score: float, expected_score: float) -> float:
    return abs(summary_score - expected_score)


def _token_counts(text: str) -> Counter[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    filtered = [token for token in tokens if token not in STOPWORDS]
    return Counter(filtered)


def cosine_similarity_texts(text_a: str, text_b: str) -> float:
    counts_a = _token_counts(text_a)
    counts_b = _token_counts(text_b)

    if not counts_a or not counts_b:
        return 0.0

    shared_tokens = counts_a.keys() & counts_b.keys()
    dot_product = sum(counts_a[token] * counts_b[token] for token in shared_tokens)
    norm_a = math.sqrt(sum(count * count for count in counts_a.values()))
    norm_b = math.sqrt(sum(count * count for count in counts_b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return dot_product / (norm_a * norm_b)


def extract_position_block_text(
    concatenated_text: str,
    dataset_position: str,
    positive_count: int | None = None,
    negative_count: int | None = None,
) -> str:
    reviews = split_reviews(concatenated_text)
    if not reviews:
        return ""

    if positive_count is not None and negative_count is not None:
        total_count = positive_count + negative_count
        if len(reviews) >= total_count and positive_count >= 0 and negative_count > 0:
            if dataset_position == "top":
                selected_reviews = reviews[:negative_count]
            elif dataset_position == "middle":
                split_point = positive_count // 2
                selected_reviews = reviews[split_point : split_point + negative_count]
            elif dataset_position == "end":
                selected_reviews = reviews[-negative_count:]
            else:
                raise ValueError(f"Unknown dataset position '{dataset_position}'.")
            return "\n\n".join(selected_reviews)

    if dataset_position == "top":
        return reviews[0]
    if dataset_position == "middle":
        return reviews[len(reviews) // 2]
    if dataset_position == "end":
        return reviews[-1]
    raise ValueError(f"Unknown dataset position '{dataset_position}'.")


def build_geval_prompt(
    source_reviews: list[str],
    summary_text: str,
    expected_score: float,
    summary_score: float,
    dataset_position: str,
    strategy: str,
) -> str:
    review_block = "\n\n".join(
        f"Review {index + 1}: {review}"
        for index, review in enumerate(source_reviews)
    )
    return (
        "You are evaluating a summary of customer reviews for a Yelp business.\n"
        f"Summarization strategy: {strategy}\n"
        f"Review position variant: {dataset_position}\n"
        "The source set contains mostly positive reviews and a minority of negative reviews.\n\n"
        f"Source Reviews:\n{review_block}\n\n"
        f"Generated Summary:\n{summary_text}\n\n"
        "Rate the summary on these criteria (1-5 scale):\n"
        "1. Negative Review Representation: Does the summary acknowledge the negative review's specific complaints? "
        "(1=completely ignored, 5=fully represented)\n"
        "2. Sentiment Proportionality: Does the summary accurately reflect that the vast majority of reviews are positive "
        "while also noting minority negative opinion? (1=misrepresents ratio, 5=perfectly proportional)\n"
        "3. Consistency: Does the summary only contain information from the source reviews, with no fabricated details? "
        "(1=many fabrications, 5=fully consistent)\n"
        "4. Coherence: Is the summary well-structured and readable? (1=incoherent, 5=excellent)\n\n"
        "Return ONLY valid JSON in this exact schema:\n"
        '{"negative_review_representation": <1-5>, "sentiment_proportionality": <1-5>, '
        '"consistency": <1-5>, "coherence": <1-5>, "overall": <1-5>}'
    )


def parse_geval_score(response_text: str) -> float:
    # Prefer an explicitly labeled overall score when the judge returns structured output.
    overall_match = re.search(
        r'"overall"\s*:\s*([1-5](?:\.\d+)?)|overall\s*[:=]\s*([1-5](?:\.\d+)?)',
        response_text,
        flags=re.IGNORECASE,
    )
    if overall_match:
        score = float(overall_match.group(1) or overall_match.group(2))
        return max(1.0, min(5.0, score))

    # Fallback: average the 4 rubric criteria if present.
    criterion_values = re.findall(
        r'"(?:negative_review_representation|sentiment_proportionality|consistency|coherence)"\s*:\s*([1-5](?:\.\d+)?)',
        response_text,
        flags=re.IGNORECASE,
    )
    if criterion_values:
        values = [float(value) for value in criterion_values]
        score = sum(values) / len(values)
        return max(1.0, min(5.0, score))

    # Last fallback for plain numeric-only outputs.
    match = re.search(r"(?<!\d)([1-5](?:\.\d+)?)(?!\d)", response_text)
    if not match:
        raise ValueError(f"Could not parse a G-Eval score from: {response_text!r}")
    score = float(match.group(1))
    return max(1.0, min(5.0, score))


def output_length_words(text: str) -> int:
    return len(re.findall(r"\S+", text))

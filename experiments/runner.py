from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .evaluation import (
    build_geval_prompt,
    cosine_similarity_texts,
    expected_aggregate_sentiment,
    extract_position_block_text,
    output_length_words,
    parse_geval_score,
    score_reviews_vader,
    score_text_vader,
    sentiment_deviation,
    split_reviews,
)
from .model_client import BaseModelClient
from .prompting import STRATEGIES, build_prompt


@dataclass
class RunConfig:
    model: str
    temperature: float
    max_tokens: int
    sample_size: int
    seed: int
    output_dir: Path
    evaluator_model: str
    evaluator_temperature: float
    evaluator_max_tokens: int
    enable_geval: bool = True


def load_dataset_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_rows(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def _int_value(row: dict, key: str, default: int | None = None) -> int | None:
    value = row.get(key)
    if value in {None, ""}:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value, default: float | None = None) -> float | None:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _average(values: list[float]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def run_experiments(
    client: BaseModelClient,
    evaluator_client: BaseModelClient | None,
    datasets: dict[str, Path],
    config: RunConfig,
) -> tuple[Path, Path, Path, list[dict], list[dict]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / f"experiment_logs_{timestamp}.jsonl"
    detailed_path = config.output_dir / f"experiment_detailed_{timestamp}.csv"
    summary_path = config.output_dir / f"experiment_summary_{timestamp}.csv"

    records: list[dict] = []

    with log_path.open("w", encoding="utf-8") as log_handle:
        for dataset_position, path in datasets.items():
            rows = load_dataset_rows(path)
            sampled = sample_rows(rows, config.sample_size, config.seed)

            for strategy in STRATEGIES:
                for row in sampled:
                    source_text = row["concatenated_text"]
                    source_reviews = split_reviews(source_text)
                    review_scores = score_reviews_vader(source_reviews)
                    expected_score = expected_aggregate_sentiment(review_scores)
                    prompt = build_prompt(strategy, source_text)

                    start = time.perf_counter()
                    result = client.generate(
                        prompt=prompt,
                        model=config.model,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    )
                    latency_s = time.perf_counter() - start

                    summary_text = (result.text or "").strip()
                    summary_score = score_text_vader(summary_text)
                    deviation = sentiment_deviation(summary_score, expected_score)

                    positive_count = _int_value(row, "positive_review_count")
                    negative_count = _int_value(row, "negative_review_count")
                    negative_block_text = extract_position_block_text(
                        source_text,
                        dataset_position,
                        positive_count=positive_count,
                        negative_count=negative_count,
                    )
                    cosine_similarity = cosine_similarity_texts(negative_block_text, summary_text)

                    geval_score = None
                    geval_response = ""
                    if config.enable_geval and evaluator_client is not None:
                        geval_prompt = build_geval_prompt(
                            source_reviews=source_reviews,
                            summary_text=summary_text,
                            expected_score=expected_score,
                            summary_score=summary_score,
                            dataset_position=dataset_position,
                            strategy=strategy,
                        )
                        geval_result = evaluator_client.generate(
                            prompt=geval_prompt,
                            model=config.evaluator_model,
                            temperature=config.evaluator_temperature,
                            max_tokens=config.evaluator_max_tokens,
                        )
                        geval_response = (geval_result.text or "").strip()
                        try:
                            geval_score = parse_geval_score(geval_response)
                        except ValueError:
                            geval_score = None

                    input_review_count = len(source_reviews)
                    output_word_count = output_length_words(summary_text)

                    record = {
                        "business_id": row["business_id"],
                        "business_name": row.get("business_name", ""),
                        "dataset_position": dataset_position,
                        "strategy": strategy,
                        "model": config.model,
                        "evaluator_model": config.evaluator_model if config.enable_geval else "",
                        "input_concatenated_text": source_text,
                        "input_review_count": input_review_count,
                        "positive_review_count": positive_count if positive_count is not None else "",
                        "negative_review_count": negative_count if negative_count is not None else "",
                        "review_scores": "|".join(f"{score:.4f}" for score in review_scores),
                        "expected_vader_score": round(expected_score, 4),
                        "summary_vader_score": round(summary_score, 4),
                        "sentiment_deviation": round(deviation, 4),
                        "negative_review_cosine_similarity": round(cosine_similarity, 4),
                        "geval_score": round(geval_score, 4) if geval_score is not None else "",
                        "geval_response": geval_response,
                        "latency_s": round(latency_s, 4),
                        "output_words": output_word_count,
                        "summary": summary_text,
                    }
                    records.append(record)

                    lean_log = {
                        "business_id": record["business_id"],
                        "dataset_position": record["dataset_position"],
                        "strategy": record["strategy"],
                        "expected_vader_score": record["expected_vader_score"],
                        "summary_vader_score": record["summary_vader_score"],
                        "sentiment_deviation": record["sentiment_deviation"],
                        "negative_review_cosine_similarity": record["negative_review_cosine_similarity"],
                        "geval_score": record["geval_score"],
                        "latency_s": record["latency_s"],
                    }
                    log_handle.write(json.dumps(lean_log, ensure_ascii=False) + "\n")

    summary_rows = aggregate_metrics(records)
    write_detailed_csv(records, detailed_path)
    write_summary_csv(summary_rows, summary_path)
    return log_path, detailed_path, summary_path, records, summary_rows


def aggregate_metrics(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        key = (record["strategy"], record["dataset_position"])
        grouped.setdefault(key, []).append(record)

    summary_rows: list[dict] = []
    for (strategy, dataset_position), rows in sorted(grouped.items()):
        n = len(rows)
        avg_expected = _average([_float_value(row.get("expected_vader_score")) for row in rows])
        avg_summary = _average([_float_value(row.get("summary_vader_score")) for row in rows])
        avg_deviation = _average([_float_value(row.get("sentiment_deviation")) for row in rows])
        avg_cosine = _average([_float_value(row.get("negative_review_cosine_similarity")) for row in rows])
        avg_geval = _average([_float_value(row.get("geval_score")) for row in rows])
        avg_latency = _average([_float_value(row.get("latency_s")) for row in rows])
        avg_len = _average([_float_value(row.get("output_words")) for row in rows])

        summary_rows.append(
            {
                "strategy": strategy,
                "dataset_position": dataset_position,
                "samples": n,
                "avg_expected_vader_score": round(avg_expected, 4) if avg_expected is not None else "",
                "avg_summary_vader_score": round(avg_summary, 4) if avg_summary is not None else "",
                "avg_sentiment_deviation": round(avg_deviation, 4) if avg_deviation is not None else "",
                "avg_negative_review_cosine_similarity": round(avg_cosine, 4) if avg_cosine is not None else "",
                "avg_geval_score": round(avg_geval, 4) if avg_geval is not None else "",
                "avg_latency_s": round(avg_latency, 4) if avg_latency is not None else "",
                "avg_output_words": round(avg_len, 2) if avg_len is not None else "",
            }
        )

    deviation_by_strategy: dict[str, list[float]] = {}
    for row in summary_rows:
        value = _float_value(row.get("avg_sentiment_deviation"))
        if value is not None:
            deviation_by_strategy.setdefault(row["strategy"], []).append(value)

    for strategy, deviations in deviation_by_strategy.items():
        spread = max(deviations) - min(deviations) if deviations else 0.0
        summary_rows.append(
            {
                "strategy": strategy,
                "dataset_position": "ALL",
                "samples": "-",
                "avg_expected_vader_score": "-",
                "avg_summary_vader_score": "-",
                "avg_sentiment_deviation": "-",
                "avg_negative_review_cosine_similarity": "-",
                "avg_geval_score": "-",
                "avg_latency_s": "-",
                "avg_output_words": "-",
                "position_sensitivity": round(spread, 4),
            }
        )

    return summary_rows


def write_summary_csv(rows: list[dict], output_path: Path):
    fieldnames = [
        "strategy",
        "dataset_position",
        "samples",
        "avg_expected_vader_score",
        "avg_summary_vader_score",
        "avg_sentiment_deviation",
        "avg_negative_review_cosine_similarity",
        "avg_geval_score",
        "avg_latency_s",
        "avg_output_words",
        "position_sensitivity",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if "position_sensitivity" not in row:
                row = {**row, "position_sensitivity": ""}
            writer.writerow(row)


def write_detailed_csv(rows: list[dict], output_path: Path):
    fieldnames = [
        "business_id",
        "business_name",
        "dataset_position",
        "strategy",
        "model",
        "evaluator_model",
        "input_review_count",
        "positive_review_count",
        "negative_review_count",
        "input_concatenated_text",
        "review_scores",
        "expected_vader_score",
        "summary_vader_score",
        "sentiment_deviation",
        "negative_review_cosine_similarity",
        "geval_score",
        "geval_response",
        "latency_s",
        "output_words",
        "summary",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

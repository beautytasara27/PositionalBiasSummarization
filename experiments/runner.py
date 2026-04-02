from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .evaluation import compression_ratio, get_negative_review_by_position, minority_recall, output_length_words
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


def load_dataset_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_rows(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def run_experiments(
    client: BaseModelClient,
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
                    negative_review = get_negative_review_by_position(source_text, dataset_position)
                    recall, keyword_recall, matched_keywords = minority_recall(summary_text, negative_review)
                    out_len_words = output_length_words(summary_text)
                    comp_ratio = compression_ratio(source_text, summary_text)

                    record = {
                        "business_id": row["business_id"],
                        "business_name": row.get("business_name", ""),
                        "dataset_position": dataset_position,
                        "strategy": strategy,
                        "model": config.model,
                        "input_concatenated_text": source_text,
                        "minority_recall": recall,
                        "keyword_recall": keyword_recall,
                        "latency_s": round(latency_s, 4),
                        "output_words": out_len_words,
                        "compression_ratio": round(comp_ratio, 4),
                        "matched_negative_keywords": "|".join(matched_keywords[:5]),
                        "summary": summary_text,
                    }
                    records.append(record)

                    lean_log = {
                        "business_id": record["business_id"],
                        "dataset_position": record["dataset_position"],
                        "strategy": record["strategy"],
                        "minority_recall": record["minority_recall"],
                        "latency_s": record["latency_s"],
                        "output_words": record["output_words"],
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

    summary_rows = []
    for (strategy, dataset_position), rows in sorted(grouped.items()):
        n = len(rows)
        avg_recall = sum(r["minority_recall"] for r in rows) / n if n else 0.0
        avg_latency = sum(r["latency_s"] for r in rows) / n if n else 0.0
        avg_len = sum(r["output_words"] for r in rows) / n if n else 0.0

        summary_rows.append(
            {
                "strategy": strategy,
                "dataset_position": dataset_position,
                "samples": n,
                "minority_recall_rate": round(avg_recall, 4),
                "avg_latency_s": round(avg_latency, 4),
                "avg_output_words": round(avg_len, 2),
            }
        )

    # Position sensitivity per strategy: max recall difference across top/middle/end.
    recalls_by_strategy: dict[str, list[float]] = {}
    for row in summary_rows:
        recalls_by_strategy.setdefault(row["strategy"], []).append(row["minority_recall_rate"])

    for strategy, recalls in recalls_by_strategy.items():
        if recalls:
            sensitivity = max(recalls) - min(recalls)
        else:
            sensitivity = 0.0
        summary_rows.append(
            {
                "strategy": strategy,
                "dataset_position": "ALL",
                "samples": "-",
                "minority_recall_rate": "-",
                "avg_latency_s": "-",
                "avg_output_words": "-",
                "position_sensitivity": round(sensitivity, 4),
            }
        )

    return summary_rows


def write_summary_csv(rows: list[dict], output_path: Path):
    fieldnames = [
        "strategy",
        "dataset_position",
        "samples",
        "minority_recall_rate",
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
        "input_concatenated_text",
        "summary",
        "minority_recall",
        "keyword_recall",
        "latency_s",
        "output_words",
        "compression_ratio",
        "matched_negative_keywords",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

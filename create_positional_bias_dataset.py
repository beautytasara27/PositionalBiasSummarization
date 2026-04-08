import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


POSITIVE_MIN_STARS = 5
NEGATIVE_MAX_STARS = 2


def load_business_metadata(path: Path):
    metadata = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metadata[record["business_id"]] = {
                "business_id": record["business_id"],
                "business_name": record.get("name", ""),
                "city": record.get("city", ""),
                "state": record.get("state", ""),
                "categories": record.get("categories", ""),
            }
    return metadata


def load_eligible_business_ids(stats_csv_path: Path):
    eligible = []
    with stats_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("meets_10_five_star_1_negative") not in {"True", "true", "1"}:
                continue
            eligible.append(
                {
                    "business_id": row["business_id"],
                    "business_name": row.get("name", ""),
                    "review_count": int(row.get("review_count") or 0),
                    "five_star_reviews": int(row.get("five_star_reviews") or 0),
                    "positive_reviews": int(row.get("positive_reviews") or 0),
                    "negative_reviews": int(row.get("negative_reviews") or 0),
                }
            )
    return eligible


def classify_review(stars: float):
    if stars >= POSITIVE_MIN_STARS:
        return "positive"
    if stars <= NEGATIVE_MAX_STARS:
        return "negative"
    return "neutral"


def collect_reviews_for_businesses(review_path: Path, selected_business_ids: set[str]):
    collected = defaultdict(lambda: {"positive": [], "negative": []})

    with review_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            business_id = record["business_id"]
            if business_id not in selected_business_ids:
                continue

            label = classify_review(float(record["stars"]))
            if label in {"positive", "negative"}:
                collected[business_id][label].append(record)

    return collected


def build_dataset_rows(selected_businesses, collected_reviews, seed: int):
    rng = random.Random(seed)
    rows = []

    for business in selected_businesses:
        business_id = business["business_id"]
        positive_reviews = collected_reviews[business_id]["positive"]
        negative_reviews = collected_reviews[business_id]["negative"]

        if len(positive_reviews) < 10:
            raise ValueError(f"Business {business_id} has only {len(positive_reviews)} positive reviews available.")
        if len(negative_reviews) < 1:
            raise ValueError(f"Business {business_id} has no negative reviews available.")

        rng.shuffle(positive_reviews)
        rng.shuffle(negative_reviews)

        sampled_reviews = positive_reviews[:10] + negative_reviews[:1]
        rng.shuffle(sampled_reviews)

        for position, review in enumerate(sampled_reviews, start=1):
            stars = float(review["stars"])
            sentiment = "positive" if stars >= POSITIVE_MIN_STARS else "negative"
            rows.append(
                {
                    "sample_id": f"{business_id}-{position}",
                    "business_id": business_id,
                    "business_name": business.get("business_name", ""),
                    "city": business.get("city", ""),
                    "state": business.get("state", ""),
                    "categories": business.get("categories", ""),
                    "review_id": review.get("review_id", ""),
                    "user_id": review.get("user_id", ""),
                    "review_text": review.get("text", ""),
                    "stars": stars,
                    "date": review.get("date", ""),
                    "useful": review.get("useful", 0),
                    "funny": review.get("funny", 0),
                    "cool": review.get("cool", 0),
                    "sentiment": sentiment,
                    "position_in_sample": position,
                }
            )

    return rows


def write_dataset(rows, output_path: Path):
    fieldnames = [
        "sample_id",
        "business_id",
        "business_name",
        "city",
        "state",
        "categories",
        "review_id",
        "user_id",
        "review_text",
        "stars",
        "date",
        "useful",
        "funny",
        "cool",
        "sentiment",
        "position_in_sample",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Create a positional-bias review dataset with 100 businesses.")
    parser.add_argument("--business-file", default="business.json", help="Path to business.json")
    parser.add_argument("--review-file", default="review.json", help="Path to review.json")
    parser.add_argument("--stats-file", default="business_review_stats.csv", help="Path to the business stats CSV")
    parser.add_argument("--output-csv", default="positional_bias_dataset.csv", help="Path for the generated dataset CSV")
    parser.add_argument("--business-limit", type=int, default=100, help="Number of businesses to include")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    business_path = (workspace / args.business_file).resolve()
    review_path = (workspace / args.review_file).resolve()
    stats_path = (workspace / args.stats_file).resolve()
    output_path = (workspace / args.output_csv).resolve()

    metadata = load_business_metadata(business_path)
    eligible_businesses = load_eligible_business_ids(stats_path)

    if len(eligible_businesses) < args.business_limit:
        raise ValueError(
            f"Only {len(eligible_businesses)} businesses meet the 10-five-star/1-negative requirement; "
            f"need {args.business_limit}."
        )

    rng = random.Random(args.seed)
    selected_businesses = rng.sample(eligible_businesses, args.business_limit)
    selected_business_ids = {business["business_id"] for business in selected_businesses}

    for business in selected_businesses:
        business_id = business["business_id"]
        business.update(metadata.get(business_id, {}))

    collected_reviews = collect_reviews_for_businesses(review_path, selected_business_ids)
    dataset_rows = build_dataset_rows(selected_businesses, collected_reviews, args.seed)
    write_dataset(dataset_rows, output_path)

    per_business_counts = defaultdict(lambda: {"five_star": 0, "negative": 0})
    for row in dataset_rows:
        key = "five_star" if row["sentiment"] == "positive" else "negative"
        per_business_counts[row["business_id"]][key] += 1

    print("Positional bias dataset created")
    print(f"Businesses meeting the requirement: {len(eligible_businesses)}")
    print(f"Businesses selected: {len(selected_businesses)}")
    print(f"Total rows written: {len(dataset_rows)}")
    print(f"Expected rows: {len(selected_businesses) * 11}")
    print(f"Output file: {output_path}")
    print()
    print("Sampled business check")
    for business in selected_businesses[:10]:
        counts = per_business_counts[business["business_id"]]
        print(
            f"- {business.get('business_name') or business['business_id']}: "
            f"five_star={counts['five_star']}, negative={counts['negative']}"
        )


if __name__ == "__main__":
    main()
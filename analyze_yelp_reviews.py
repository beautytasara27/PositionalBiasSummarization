import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def load_business_metadata(path: Path):
    metadata = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metadata[record["business_id"]] = {
                "name": record.get("name", ""),
                "reported_review_count": record.get("review_count"),
            }
    return metadata


def classify_review(stars: float):
    if stars >= 4:
        return "positive"
    if stars <= 2:
        return "negative"
    return "neutral"


def analyze_reviews(path: Path):
    counts = defaultdict(lambda: {"total": 0, "five_star": 0, "positive": 0, "negative": 0, "neutral": 0})
    total_reviews = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            business_id = record["business_id"]
            stars = float(record["stars"])
            label = classify_review(stars)

            counts[business_id]["total"] += 1
            if stars == 5:
                counts[business_id]["five_star"] += 1
            counts[business_id][label] += 1
            total_reviews += 1

    return counts, total_reviews


def build_rows(business_metadata, review_counts):
    rows = []
    all_business_ids = set(business_metadata) | set(review_counts)

    for business_id in all_business_ids:
        meta = business_metadata.get(business_id, {})
        counts = review_counts.get(business_id, {"total": 0, "positive": 0, "negative": 0, "neutral": 0})
        total = counts["total"]
        five_star = counts.get("five_star", 0)
        positive = counts["positive"]
        negative = counts["negative"]
        neutral = counts["neutral"]
        meets_threshold = five_star >= 10 and negative >= 1

        rows.append(
            {
                "business_id": business_id,
                "name": meta.get("name", ""),
                "reported_review_count": meta.get("reported_review_count", ""),
                "review_count": total,
                "five_star_reviews": five_star,
                "positive_reviews": positive,
                "negative_reviews": negative,
                "neutral_reviews": neutral,
                "meets_10_five_star_1_negative": meets_threshold,
            }
        )

    rows.sort(key=lambda row: (row["review_count"], row["five_star_reviews"], row["negative_reviews"]), reverse=True)
    return rows


def write_csv(rows, output_path: Path):
    fieldnames = [
        "business_id",
        "name",
        "reported_review_count",
        "review_count",
        "five_star_reviews",
        "positive_reviews",
        "negative_reviews",
        "neutral_reviews",
        "meets_10_five_star_1_negative",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Analyze Yelp business and review datasets.")
    parser.add_argument("--business-file", default="business.json", help="Path to business.json")
    parser.add_argument("--review-file", default="review.json", help="Path to review.json")
    parser.add_argument("--output-csv", default="business_review_stats.csv", help="Path for the detailed CSV output")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    business_path = (workspace / args.business_file).resolve()
    review_path = (workspace / args.review_file).resolve()
    output_path = (workspace / args.output_csv).resolve()

    business_metadata = load_business_metadata(business_path)
    review_counts, total_reviews = analyze_reviews(review_path)
    rows = build_rows(business_metadata, review_counts)

    reviewed_business_rows = [row for row in rows if row["review_count"] > 0]
    threshold_rows = [row for row in reviewed_business_rows if row["meets_10_five_star_1_negative"]]
    review_counts_only = [row["review_count"] for row in reviewed_business_rows]

    write_csv(rows, output_path)

    total_businesses = len(rows)
    reviewed_businesses = len(reviewed_business_rows)
    threshold_businesses = len(threshold_rows)

    print("Dataset summary")
    print(f"Businesses in business.json: {total_businesses}")
    print(f"Businesses with at least one review: {reviewed_businesses}")
    print(f"Total reviews in review.json: {total_reviews}")
    print(f"Five-star reviews: {sum(row['five_star_reviews'] for row in reviewed_business_rows)}")
    print(f"Positive reviews (4-5 stars): {sum(row['positive_reviews'] for row in reviewed_business_rows)}")
    print(f"Negative reviews (1-2 stars): {sum(row['negative_reviews'] for row in reviewed_business_rows)}")
    print(f"Neutral reviews (3 stars): {sum(row['neutral_reviews'] for row in reviewed_business_rows)}")
    print()
    print("Per-business review count stats")
    print(f"Minimum reviews: {min(review_counts_only) if review_counts_only else 0}")
    print(f"Median reviews: {median(review_counts_only) if review_counts_only else 0}")
    print(f"Average reviews: {mean(review_counts_only):.2f}" if review_counts_only else "Average reviews: 0")
    print(f"Maximum reviews: {max(review_counts_only) if review_counts_only else 0}")
    print()
    print("Threshold check")
    print(f"Businesses with at least 10 five-star reviews and at least 1 negative review: {threshold_businesses}")
    if reviewed_businesses:
        print(f"Share of reviewed businesses meeting threshold: {threshold_businesses / reviewed_businesses:.2%}")
    print()
    print(f"Detailed CSV written to: {output_path}")
    print()
    print("Top 10 businesses by review count")
    for row in reviewed_business_rows[:10]:
        print(
            f"- {row['name'] or row['business_id']}: total={row['review_count']}, "
            f"five_star={row['five_star_reviews']}, negative={row['negative_reviews']}, "
            f"meets_threshold={row['meets_10_five_star_1_negative']}"
        )


if __name__ == "__main__":
    main()
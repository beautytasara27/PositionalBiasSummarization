import argparse
import csv
from collections import defaultdict
from pathlib import Path


POSITIONS = ("top", "middle", "end")


def load_rows(path: Path):
    grouped = defaultdict(list)
    business_names = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            business_id = row["business_id"]
            grouped[business_id].append(row)
            business_names[business_id] = row.get("business_name", "")

    return grouped, business_names


def ordered_reviews(rows, negative_position: str):
    positive_reviews = [row for row in rows if row.get("sentiment") == "positive"]
    negative_reviews = [row for row in rows if row.get("sentiment") == "negative"]

    if len(positive_reviews) != 10:
        raise ValueError(f"Expected 10 positive reviews, found {len(positive_reviews)}")
    if len(negative_reviews) != 1:
        raise ValueError(f"Expected 1 negative review, found {len(negative_reviews)}")

    negative_review = negative_reviews[0]

    if negative_position == "top":
        ordered = [negative_review] + positive_reviews
    elif negative_position == "middle":
        split_point = len(positive_reviews) // 2
        ordered = positive_reviews[:split_point] + [negative_review] + positive_reviews[split_point:]
    elif negative_position == "end":
        ordered = positive_reviews + [negative_review]
    else:
        raise ValueError(f"Unsupported negative position: {negative_position}")

    return ordered


def build_concatenated_rows(grouped_rows, business_names, negative_position: str):
    dataset_rows = []

    for business_id, rows in grouped_rows.items():
        ordered = ordered_reviews(rows, negative_position)
        concatenated_text = "\n\n".join(row["review_text"].strip() for row in ordered)
        dataset_rows.append(
            {
                "business_id": business_id,
                "business_name": business_names.get(business_id, ""),
                "concatenated_text": concatenated_text,
            }
        )

    dataset_rows.sort(key=lambda row: row["business_name"] or row["business_id"])
    return dataset_rows


def write_csv(rows, output_path: Path):
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["business_id", "business_name", "concatenated_text"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Create per-business concatenated review datasets with one negative review placed at different positions.")
    parser.add_argument("--input-csv", default="positional_bias_dataset.csv", help="Path to the 1100-row sample dataset CSV")
    parser.add_argument("--output-prefix", default="positional_bias_concatenated", help="Prefix for the generated CSV files")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    input_path = (workspace / args.input_csv).resolve()

    grouped_rows, business_names = load_rows(input_path)

    for position in POSITIONS:
        output_path = (workspace / f"{args.output_prefix}_{position}.csv").resolve()
        rows = build_concatenated_rows(grouped_rows, business_names, position)
        write_csv(rows, output_path)
        print(f"Wrote {len(rows)} businesses to {output_path}")


if __name__ == "__main__":
    main()
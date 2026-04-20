"""
compute_bertscore.py
---------------------
Drop into repo root and run:

    python compute_bertscore.py --input-dir experiment_outputs

Adds BERTScore columns to your existing experiment_detailed_*.csv files
and saves new bertscore_*.csv files alongside them.

WHY BERTScore > cosine similarity
----------------------------------
Your current negative_review_cosine_similarity uses TF-IDF / raw vectors,
which misses paraphrases ("scalp irritation" != "bloody scabs" to cosine,
but BERTScore catches them as semantically close). BERTScore F1 correlates
much better with human judgments of summary faithfulness.

Columns added
-------------
  bertscore_neg_f1        semantic similarity: summary vs negative review text
  bertscore_neg_precision
  bertscore_neg_recall
  bertscore_all_f1        semantic similarity: summary vs ALL input reviews
  neg_attention_ratio     bertscore_neg_f1 / bertscore_all_f1
                          -- stable across positions = no bias
                          -- drops at middle = "lost in the middle" confirmed

Column source mapping (your actual CSV columns)
-----------------------------------------------
  summary                 <- "summary" column
  negative review text    <- extracted from "input_concatenated_text"
                             (first review if position=top,
                              middle chunk if position=middle,
                              last review if position=end)
  all reviews text        <- "input_concatenated_text" (full)

NOTE: If your CSV has a dedicated "negative_review_text" column already,
set --neg-col to that column name and we'll use it directly.

Install once:
    pip install bert-score
    # Downloads roberta-large on first run (~1.4 GB, cached after that)
    # Use --model distilbert-base-uncased for a faster/smaller alternative
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd

# ── column names matched to your actual CSVs ──────────────────────────────────

SUMMARY_COL       = "summary"
FULL_INPUT_COL    = "input_concatenated_text"
POSITION_COL      = "dataset_position"
STRATEGY_COL      = "strategy"
SCALE_COL         = "positive_review_count"

# ── BERTScore wrapper ─────────────────────────────────────────────────────────

def _get_bert_score():
    try:
        from bert_score import score
        return score
    except ImportError:
        raise ImportError(
            "bert-score not installed.\nRun:  pip install bert-score"
        )

def compute_bertscore(
    hypotheses: list[str],
    references: list[str],
    model_type: str = "roberta-large",
    batch_size: int = 16,
) -> tuple[list[float], list[float], list[float]]:
    score = _get_bert_score()
    P, R, F = score(
        cands=hypotheses,
        refs=references,
        model_type=model_type,
        batch_size=batch_size,
        lang="en",
        verbose=True,
    )
    return P.tolist(), R.tolist(), F.tolist()

# ── negative review extractor ─────────────────────────────────────────────────

def extract_negative_review(row: pd.Series, neg_col: str | None) -> str:
    """
    Get just the negative review text for a row.

    Priority:
    1. Use dedicated neg_col if provided and non-empty
    2. Otherwise extract from input_concatenated_text based on position:
       - top    -> first review block
       - end    -> last review block
       - middle -> middle review block
    Reviews in your concatenated text appear to be separated by newlines.
    """
    if neg_col and neg_col in row.index and pd.notna(row[neg_col]) and str(row[neg_col]).strip():
        return str(row[neg_col]).strip()

    full_text = str(row.get(FULL_INPUT_COL, "")).strip()
    if not full_text:
        return ""

    # Split on double newline (common separator between reviews)
    blocks = [b.strip() for b in full_text.split("\n\n") if b.strip()]
    if not blocks:
        blocks = [b.strip() for b in full_text.split("\n") if b.strip()]

    position = str(row.get(POSITION_COL, "")).lower()
    if position == "top" and blocks:
        return blocks[0]
    elif position == "end" and blocks:
        return blocks[-1]
    elif position == "middle" and blocks:
        return blocks[len(blocks) // 2]
    elif blocks:
        return blocks[0]
    return full_text[:500]   # fallback: first 500 chars

# ── main scoring function ─────────────────────────────────────────────────────

def score_dataframe(
    df: pd.DataFrame,
    model_type: str = "roberta-large",
    neg_col: str | None = None,
) -> pd.DataFrame:
    df = df.copy()

    for col in [SUMMARY_COL, FULL_INPUT_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found.\n"
                f"Available columns: {list(df.columns)}"
            )

    summaries   = df[SUMMARY_COL].fillna("").tolist()
    full_inputs = df[FULL_INPUT_COL].fillna("").tolist()
    neg_reviews = [extract_negative_review(row, neg_col) for _, row in df.iterrows()]

    # Score 1: summary vs negative review
    print(f"\n  BERTScore: summary vs negative review ({len(summaries)} samples)...")
    P_neg, R_neg, F_neg = compute_bertscore(summaries, neg_reviews, model_type=model_type)
    df["bertscore_neg_precision"] = [round(v, 5) for v in P_neg]
    df["bertscore_neg_recall"]    = [round(v, 5) for v in R_neg]
    df["bertscore_neg_f1"]        = [round(v, 5) for v in F_neg]

    # Score 2: summary vs all input reviews
    print(f"  BERTScore: summary vs all input reviews ({len(summaries)} samples)...")
    _, _, F_all = compute_bertscore(summaries, full_inputs, model_type=model_type)
    df["bertscore_all_f1"] = [round(v, 5) for v in F_all]

    # Ratio: if model attends equally regardless of position, this should be flat
    df["neg_attention_ratio"] = [
        round(neg / all_f if all_f > 0 else float("nan"), 5)
        for neg, all_f in zip(F_neg, F_all)
    ]

    return df

# ── summary table ─────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in [STRATEGY_COL, POSITION_COL, SCALE_COL] if c in df.columns]
    metric_cols = [c for c in ["bertscore_neg_f1", "bertscore_all_f1", "neg_attention_ratio"] if c in df.columns]
    if not group_cols or not metric_cols:
        return pd.DataFrame()
    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).round(5)
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()

# ── standalone CLI ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Add BERTScore columns to experiment detailed CSVs.")
    p.add_argument("--input-dir",   default="experiment_outputs")
    p.add_argument("--output-dir",  default="experiment_outputs")
    p.add_argument("--model-type",  default="roberta-large",
                   help="HuggingFace model. Use 'distilbert-base-uncased' for faster/smaller.")
    p.add_argument("--neg-col",     default=None,
                   help="Column name of the dedicated negative review text, if it exists in your CSV.")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "experiment_detailed_*.csv")))
    if not files:
        print(f"No experiment_detailed_*.csv found in {args.input_dir!r}. Exiting.")
        return

    all_scored = []
    for fpath in files:
        print(f"\nProcessing: {fpath}")
        df = pd.read_csv(fpath)
        scored = score_dataframe(df, model_type=args.model_type, neg_col=args.neg_col)
        out_path = os.path.join(args.output_dir, f"bertscore_{Path(fpath).name}")
        scored.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")
        all_scored.append(scored)

    combined     = pd.concat(all_scored, ignore_index=True)
    summary      = summarize(combined)
    summary_path = os.path.join(args.output_dir, "bertscore_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nBERTScore summary -> {summary_path}")
    print("\nPreview (mean neg_f1 by position and strategy):")
    if "bertscore_neg_f1_mean" in summary.columns:
        cols = [c for c in [STRATEGY_COL, POSITION_COL, "bertscore_neg_f1_mean", "neg_attention_ratio_mean"] if c in summary.columns]
        print(summary[cols].to_string(index=False))

# ── single-pair convenience (import into runner.py) ───────────────────────────

def score_single(summary: str, negative_review: str, model_type: str = "roberta-large") -> dict:
    """
    Score one (summary, negative_review) pair inline.

    Usage inside experiments/runner.py:
        from compute_bertscore import score_single
        bs = score_single(summary_text, neg_review_text)
        record.update(bs)
    """
    P, R, F = compute_bertscore([summary], [negative_review], model_type=model_type)
    return {
        "bertscore_neg_precision": round(P[0], 5),
        "bertscore_neg_recall":    round(R[0], 5),
        "bertscore_neg_f1":        round(F[0], 5),
    }

if __name__ == "__main__":
    main()

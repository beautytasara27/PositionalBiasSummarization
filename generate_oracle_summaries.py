"""
generate_oracle_summaries.py
-----------------------------
Drop into repo root and run:

    python generate_oracle_summaries.py \
        --model "openai/gpt-4o" \
        --limit 10

What it does
------------
Builds a position-free "oracle" (reference) summary for each business,
which becomes your ground truth for BERTScore / cosine comparisons.

Without an oracle, you can only show that summaries DIFFER by position.
With an oracle, you can show that middle-position summaries are WORSE
(farther from the ideal) — which is a much stronger claim.

How it reads your data
-----------------------
Your positional_bias_dataset.csv has one row per review, not per business.
Columns used:
  business_id         groups rows into businesses
  business_name       for labeling
  review_text         the actual review content
  stars               1-2 = negative, 4-5 = positive (3 is excluded)
  sentiment           'positive' / 'negative' (if present, used directly)

Oracle construction (two-stage, position-free)
----------------------------------------------
Stage 1 — Theme extraction:
  ALL reviews for a business are presented in SHUFFLED order (random seed 99),
  labeled "Review 1:", "Review 2:", etc. The model extracts:
    - Key positive themes (bullet points)
    - Key negative complaints (must be specific, not softened)

Stage 2 — Faithful synthesis:
  Using only the extracted themes, the model writes a balanced 3-4 sentence
  summary that MUST name the negative complaint explicitly.

This is equivalent to what a careful human annotator would produce —
it is the upper-bound reference your positional variants are scored against.

Output
------
oracle_summaries.csv  (one row per business, new file)
  business_id
  business_name
  oracle_summary
  oracle_neg_themes
  oracle_pos_themes
  oracle_model
  oracle_error        (empty if successful)

Then use oracle_summary as the reference in compute_bertscore.py:
    python compute_bertscore.py --oracle-csv oracle_summaries.csv

Install once:
    pip install openai
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd

# ── column names matched to your actual positional_bias_dataset.csv ────────────

BUSINESS_ID_COL   = "business_id"
BUSINESS_NAME_COL = "business_name"
REVIEW_TEXT_COL   = "review_text"
STARS_COL         = "stars"
SENTIMENT_COL     = "sentiment"   # optional — used if present

# ── prompt templates ──────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """\
You are a careful research assistant analyzing customer reviews.
Below are {n} reviews for a business, in random order.

Your job is NOT to summarize yet. First, extract:
1. KEY POSITIVE THEMES: What do most positive reviews praise? (up to 5 bullet points)
2. KEY NEGATIVE COMPLAINTS: What does the negative review(s) complain about?
   Be specific. Do not soften, omit, or minimize any complaint.

Reviews:
{labeled_reviews}

Respond ONLY in this exact JSON format, no extra text:
{{
  "positive_themes": ["theme 1", "theme 2"],
  "negative_complaints": ["complaint 1", "complaint 2"]
}}"""

SYNTHESIS_PROMPT = """\
Write a 3-4 sentence customer review summary for a business.

Positive themes from reviews:
{pos_themes}

Negative complaints from reviews:
{neg_complaints}

Requirements:
- Accurately reflect the majority positive experience
- Explicitly and clearly state the negative complaint(s) — do not bury or soften them
- Do not add any information not listed above
- Use neutral, third-person voice

Summary:"""

# ── OpenRouter / OpenAI caller ────────────────────────────────────────────────

def load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

def call_model(model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    import openai
    api_key  = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = (
        "https://openrouter.ai/api/v1"
        if os.environ.get("OPENROUTER_API_KEY")
        else "https://api.openai.com/v1"
    )
    if not api_key:
        raise ValueError(
            "Set OPENROUTER_API_KEY or OPENAI_API_KEY in your .env file or environment."
        )
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    resp   = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# ── per-business oracle generation ───────────────────────────────────────────

def classify_review(row: pd.Series) -> str:
    """Return 'positive' or 'negative' for a review row."""
    if SENTIMENT_COL in row.index and pd.notna(row[SENTIMENT_COL]):
        return str(row[SENTIMENT_COL]).strip().lower()
    stars = float(row.get(STARS_COL, 0))
    if stars >= 4:
        return "positive"
    if stars <= 2:
        return "negative"
    return "neutral"

def generate_oracle_for_business(
    business_rows: pd.DataFrame,
    model: str,
    delay: float = 0.5,
) -> dict:
    biz_id   = business_rows[BUSINESS_ID_COL].iloc[0]
    biz_name = business_rows.get(BUSINESS_NAME_COL, pd.Series(["Unknown"])).iloc[0]

    pos_reviews = business_rows[business_rows.apply(classify_review, axis=1) == "positive"][REVIEW_TEXT_COL].tolist()
    neg_reviews = business_rows[business_rows.apply(classify_review, axis=1) == "negative"][REVIEW_TEXT_COL].tolist()

    if not pos_reviews:
        return {"business_id": biz_id, "business_name": biz_name, "oracle_summary": "",
                "oracle_neg_themes": "", "oracle_pos_themes": "", "oracle_model": model,
                "oracle_error": "no positive reviews found"}
    if not neg_reviews:
        return {"business_id": biz_id, "business_name": biz_name, "oracle_summary": "",
                "oracle_neg_themes": "", "oracle_pos_themes": "", "oracle_model": model,
                "oracle_error": "no negative reviews found"}

    # Shuffle all reviews so model sees no positional pattern
    all_reviews = [(t, "positive") for t in pos_reviews] + [(t, "negative") for t in neg_reviews]
    random.seed(99)
    random.shuffle(all_reviews)

    labeled = "\n\n".join(
        f"Review {i+1}: {text.strip()}"
        for i, (text, _) in enumerate(all_reviews)
    )

    # Stage 1 — extract themes
    ext_prompt = EXTRACTION_PROMPT.format(n=len(all_reviews), labeled_reviews=labeled)

    for attempt in range(2):
        try:
            raw   = call_model(model, ext_prompt, temperature=0.0, max_tokens=400)
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            themes = json.loads(clean)
            pos_themes  = themes.get("positive_themes", [])
            neg_complaints = themes.get("negative_complaints", [])
            break
        except (json.JSONDecodeError, KeyError, Exception) as e:
            if attempt == 0:
                time.sleep(delay)
            else:
                return {"business_id": biz_id, "business_name": biz_name, "oracle_summary": "",
                        "oracle_neg_themes": "", "oracle_pos_themes": "", "oracle_model": model,
                        "oracle_error": f"Stage 1 failed: {str(e)[:120]}"}

    time.sleep(delay)

    # Stage 2 — synthesize
    syn_prompt = SYNTHESIS_PROMPT.format(
        pos_themes="\n".join(f"- {t}" for t in pos_themes),
        neg_complaints="\n".join(f"- {c}" for c in neg_complaints),
    )
    try:
        oracle_summary = call_model(model, syn_prompt, temperature=0.0, max_tokens=220)
    except Exception as e:
        return {"business_id": biz_id, "business_name": biz_name, "oracle_summary": "",
                "oracle_neg_themes": " | ".join(neg_complaints),
                "oracle_pos_themes": " | ".join(pos_themes), "oracle_model": model,
                "oracle_error": f"Stage 2 failed: {str(e)[:120]}"}

    return {
        "business_id":      biz_id,
        "business_name":    biz_name,
        "oracle_summary":   oracle_summary.strip(),
        "oracle_neg_themes": " | ".join(neg_complaints),
        "oracle_pos_themes": " | ".join(pos_themes),
        "oracle_model":     model,
        "oracle_error":     "",
    }

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate oracle (ground truth) summaries from positional_bias_dataset.csv")
    p.add_argument("--input-csv",   default="positional_bias_dataset.csv",
                   help="Your base dataset (one review per row)")
    p.add_argument("--output-csv",  default="oracle_summaries.csv",
                   help="Output file (one oracle summary per business)")
    p.add_argument("--model",       default="openai/gpt-4o",
                   help="Model to use. Use a strong model — gpt-4o recommended. Runs once per business.")
    p.add_argument("--limit",       type=int, default=None,
                   help="Only process first N businesses (for testing)")
    p.add_argument("--delay",       type=float, default=0.5,
                   help="Seconds between API calls")
    return p.parse_args()

def main():
    args = parse_args()
    load_env(Path(".env"))

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df):,} rows from {args.input_csv!r}")
    print(f"Columns: {list(df.columns)}\n")

    # Validate required columns
    for col in [BUSINESS_ID_COL, REVIEW_TEXT_COL]:
        if col not in df.columns:
            print(f"ERROR: Required column '{col}' not found in {args.input_csv!r}")
            print(f"Available columns: {list(df.columns)}")
            return

    businesses = df[BUSINESS_ID_COL].unique()
    if args.limit:
        businesses = businesses[:args.limit]
        print(f"Limiting to first {args.limit} businesses (--limit flag).\n")

    results = []
    for i, biz_id in enumerate(businesses):
        biz_rows = df[df[BUSINESS_ID_COL] == biz_id]
        biz_name = biz_rows[BUSINESS_NAME_COL].iloc[0] if BUSINESS_NAME_COL in biz_rows.columns else biz_id
        n_pos = (biz_rows.apply(classify_review, axis=1) == "positive").sum()
        n_neg = (biz_rows.apply(classify_review, axis=1) == "negative").sum()

        print(f"[{i+1}/{len(businesses)}] {biz_name}  ({n_pos} pos, {n_neg} neg)")

        result = generate_oracle_for_business(biz_rows, model=args.model, delay=args.delay)
        results.append(result)

        if result["oracle_error"]:
            print(f"  WARNING: {result['oracle_error']}")
        else:
            preview = result["oracle_summary"][:110].replace("\n", " ")
            print(f"  OK: {preview}...")

        time.sleep(args.delay)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nDone. {len(out_df)} oracle summaries saved -> {args.output_csv!r}")

    errors = out_df[out_df["oracle_error"] != ""]
    if not errors.empty:
        print(f"\n{len(errors)} businesses had errors:")
        print(errors[["business_id", "oracle_error"]].to_string(index=False))

if __name__ == "__main__":
    main()

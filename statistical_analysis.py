"""
statistical_analysis.py
------------------------
Drop into repo root and run:

    python statistical_analysis.py --input-dir experiment_outputs

Reads all experiment_detailed_*.csv files and outputs:
  - experiment_outputs/statistical_analysis_results.csv
  - experiment_outputs/statistical_analysis_report.txt

Tests per (strategy x metric x scale):
  - One-way ANOVA across positions (top / middle / end)
  - Tukey HSD post-hoc pairwise comparisons
  - eta-squared effect size
  - Kruskal-Wallis non-parametric backup

Install once:
    pip install scipy statsmodels pandas
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd
from scipy import stats
from scipy.stats import kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ── column names matched to your actual experiment_detailed_*.csv ─────────────
POSITION_COL = "dataset_position"        # values: top / middle / end
STRATEGY_COL = "strategy"               # baseline / cot / repetition
SCALE_COL    = "positive_review_count"  # 10 / 20 / 30 / 40 / 50

METRICS = [
    "sentiment_deviation",
    "negative_review_cosine_similarity",
    "geval_score",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def eta_squared(f_stat: float, df_between: int, df_within: int) -> float:
    ss_between = f_stat * df_between
    ss_total   = ss_between + df_within
    return ss_between / ss_total if ss_total > 0 else float("nan")

def interpret_eta(eta: float) -> str:
    if eta < 0.01:  return "negligible"
    if eta < 0.06:  return "small"
    if eta < 0.14:  return "medium"
    return "large"

def interpret_p(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ── data loading ──────────────────────────────────────────────────────────────

def load_data(input_dir: str) -> pd.DataFrame:
    pattern = os.path.join(input_dir, "experiment_detailed_*.csv")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No experiment_detailed_*.csv files found in {input_dir!r}.\n"
            f"Run run_experiments.py first."
        )
    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        df["_source_file"] = Path(f).name
        dfs.append(df)
        print(f"  Loaded {len(df):4d} rows from {Path(f).name}")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(combined):,} rows\n")

    for col in [POSITION_COL, STRATEGY_COL]:
        if col not in combined.columns:
            raise ValueError(
                f"Required column '{col}' not found.\n"
                f"Available: {list(combined.columns)}"
            )
    return combined

# ── core ANOVA block ──────────────────────────────────────────────────────────

def run_anova_block(df: pd.DataFrame, metric: str) -> dict:
    valid_groups = {
        label: grp[metric].dropna().values
        for label, grp in df.groupby(POSITION_COL)
        if grp[metric].dropna().shape[0] >= 2
    }
    if len(valid_groups) < 2:
        return {"error": f"fewer than 2 positions with sufficient data for {metric}"}

    groups = list(valid_groups.values())
    n_total    = sum(len(g) for g in groups)
    df_between = len(groups) - 1
    df_within  = n_total - len(groups)

    f_stat, p_anova   = stats.f_oneway(*groups)
    h_stat, p_kruskal = kruskal(*groups)
    eta2              = eta_squared(f_stat, df_between, df_within)

    tukey_input = df[[POSITION_COL, metric]].dropna()
    tukey       = pairwise_tukeyhsd(
        endog=tukey_input[metric],
        groups=tukey_input[POSITION_COL],
        alpha=0.05,
    )
    tukey_rows = [
        {
            "group1":    str(row[0]),
            "group2":    str(row[1]),
            "mean_diff": round(float(row[2]), 5),
            "p_adj":     round(float(row[3]), 5),
            "lower_ci":  round(float(row[4]), 5),
            "upper_ci":  round(float(row[5]), 5),
            "reject":    bool(row[6]),
        }
        for row in tukey.summary().data[1:]
    ]

    descriptives = {
        label: {
            "n":    len(g),
            "mean": round(float(g.mean()), 5),
            "std":  round(float(g.std()),  5),
        }
        for label, g in valid_groups.items()
    }

    return {
        "f_stat":         round(f_stat,    4),
        "p_anova":        round(p_anova,   6),
        "sig_anova":      interpret_p(p_anova),
        "df_between":     df_between,
        "df_within":      df_within,
        "eta_squared":    round(eta2,      4),
        "effect_size":    interpret_eta(eta2),
        "h_stat_kruskal": round(h_stat,    4),
        "p_kruskal":      round(p_kruskal, 6),
        "sig_kruskal":    interpret_p(p_kruskal),
        "descriptives":   descriptives,
        "tukey":          tukey_rows,
    }

# ── full analysis grid ────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame) -> list[dict]:
    records   = []
    has_scale = SCALE_COL in df.columns
    strategies = sorted(df[STRATEGY_COL].unique())
    scales     = sorted(df[SCALE_COL].dropna().unique()) if has_scale else []

    for strategy in strategies:
        strat_df = df[df[STRATEGY_COL] == strategy]
        for metric in METRICS:
            if metric not in df.columns or strat_df[metric].dropna().empty:
                print(f"  Skipping {strategy} / {metric} — no data")
                continue

            # All scales pooled
            result = run_anova_block(strat_df, metric)
            records.append({"strategy": strategy, "metric": metric, "scale": "ALL", **_flatten(result)})

            # Per-scale breakdown
            if has_scale:
                for scale in scales:
                    sub = strat_df[strat_df[SCALE_COL] == scale]
                    if sub.shape[0] < 6:
                        continue
                    result = run_anova_block(sub, metric)
                    records.append({"strategy": strategy, "metric": metric, "scale": int(scale), **_flatten(result)})

    return records

def _flatten(d: dict) -> dict:
    return {k: str(v) if isinstance(v, (dict, list)) else v for k, v in d.items()}

# ── report writer ─────────────────────────────────────────────────────────────

def write_report(records: list[dict], out_path: str):
    lines = [
        "=" * 72,
        "POSITIONAL BIAS — STATISTICAL ANALYSIS REPORT",
        "=" * 72,
        "Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant",
        "Effect η²:    negligible<0.01 | small<0.06 | medium<0.14 | large>=0.14",
        "",
    ]

    by_strategy: dict[str, list] = {}
    for r in records:
        by_strategy.setdefault(r["strategy"], []).append(r)

    for strategy, rows in by_strategy.items():
        lines += ["─" * 72, f"  STRATEGY: {strategy.upper()}", "─" * 72]
        for row in rows:
            if "error" in row:
                lines.append(f"\n  [{row.get('metric','')} scale={row.get('scale','')}] ERROR: {row['error']}")
                continue
            scale_tag = f"scale={row['scale']}" if row["scale"] != "ALL" else "ALL SCALES POOLED"
            lines += [
                f"\n  Metric : {row['metric']}  [{scale_tag}]",
                f"  ANOVA  : F({row.get('df_between','?')},{row.get('df_within','?')}) = "
                f"{row.get('f_stat','?')},  p = {row.get('p_anova','?')} {row.get('sig_anova','')}",
                f"  Effect : eta2 = {row.get('eta_squared','?')}  ({row.get('effect_size','?')})",
                f"  Kruskal: H = {row.get('h_stat_kruskal','?')},  "
                f"p = {row.get('p_kruskal','?')} {row.get('sig_kruskal','')}",
            ]
            try:
                desc = eval(row["descriptives"])
                lines.append("  Group means:")
                for pos in ["top", "middle", "end"]:
                    if pos in desc:
                        v = desc[pos]
                        lines.append(f"    {pos:8s}  n={v['n']:4d}  mean={v['mean']:.5f}  std={v['std']:.5f}")
            except Exception:
                pass
            try:
                tukey = eval(row["tukey"])
                sig   = [t for t in tukey if t["reject"]]
                if sig:
                    lines.append("  Significant Tukey pairs:")
                    for t in sig:
                        lines.append(
                            f"    {t['group1']} vs {t['group2']}:  "
                            f"delta={t['mean_diff']:+.5f}  p_adj={t['p_adj']:.4f}  "
                            f"95%CI [{t['lower_ci']:.5f}, {t['upper_ci']:.5f}]"
                        )
                else:
                    lines.append("  Tukey HSD: no significant pairwise differences")
            except Exception:
                pass

    lines += ["", "=" * 72, "END OF REPORT", "=" * 72]
    text = "\n".join(lines)
    Path(out_path).write_text(text, encoding="utf-8")
    print(text)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ANOVA + Tukey HSD + effect sizes for positional bias results.")
    p.add_argument("--input-dir",  default="experiment_outputs", help="Folder with experiment_detailed_*.csv files")
    p.add_argument("--output-dir", default="experiment_outputs", help="Where to write results")
    return p.parse_args()

def main():
    args = parse_args()
    df   = load_data(args.input_dir)
    print("Running ANOVA grid...\n")
    records = analyze(df)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path    = os.path.join(args.output_dir, "statistical_analysis_results.csv")
    report_path = os.path.join(args.output_dir, "statistical_analysis_report.txt")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    write_report(records, report_path)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {report_path}")

if __name__ == "__main__":
    main()

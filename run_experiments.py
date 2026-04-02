from __future__ import annotations

import argparse
from pathlib import Path

from experiments.model_client import build_model_client
from experiments.runner import RunConfig, run_experiments


def parse_args():
    parser = argparse.ArgumentParser(description="Run positional-bias summarization experiments.")
    parser.add_argument("--provider", default="mock", choices=["mock", "openai"], help="Model provider")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name (provider-specific)")
    parser.add_argument("--sample-size", type=int, default=10, help="Samples per dataset position")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=220, help="Max output tokens")
    parser.add_argument("--top-csv", default="positional_bias_concatenated_top.csv")
    parser.add_argument("--middle-csv", default="positional_bias_concatenated_middle.csv")
    parser.add_argument("--end-csv", default="positional_bias_concatenated_end.csv")
    parser.add_argument("--output-dir", default="experiment_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    workspace = Path(__file__).resolve().parent

    datasets = {
        "top": (workspace / args.top_csv).resolve(),
        "middle": (workspace / args.middle_csv).resolve(),
        "end": (workspace / args.end_csv).resolve(),
    }

    client = build_model_client(args.provider)
    config = RunConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=(workspace / args.output_dir).resolve(),
    )

    log_path, detailed_path, summary_path, records, summary_rows = run_experiments(
        client=client,
        datasets=datasets,
        config=config,
    )

    print("Experiment run complete")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Total inferences: {len(records)}")
    print(f"Lean logs: {log_path}")
    print(f"Detailed per-sample CSV: {detailed_path}")
    print(f"Summary metrics: {summary_path}")
    print()
    print("Quick metric view")
    for row in summary_rows:
        if row["dataset_position"] == "ALL":
            print(f"- {row['strategy']}: position_sensitivity={row.get('position_sensitivity', '')}")
        else:
            print(
                f"- {row['strategy']} | {row['dataset_position']}: "
                f"recall={row['minority_recall_rate']}, "
                f"latency={row['avg_latency_s']}s, "
                f"output_words={row['avg_output_words']}"
            )


if __name__ == "__main__":
    main()

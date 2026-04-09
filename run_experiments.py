from __future__ import annotations

import argparse
import os
from pathlib import Path

from experiments.model_client import OPENROUTER_MODELS, build_model_client
from experiments.runner import RunConfig, run_experiments


def load_env_file(env_path: Path):
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args():
    parser = argparse.ArgumentParser(description="Run positional-bias summarization experiments.")
    parser.add_argument(
        "--provider",
        default="openrouter",
        choices=["mock", "openrouter", "openai"],
        help="Model provider for summary generation",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help=(
            "Model alias or full provider model name. "
            "Examples: 'Gemini 2.0 Flash', 'Claude 3.7 Sonnet', "
            "'openai/gpt-4o-mini'."
        ),
    )
    parser.add_argument(
        "--evaluator-provider",
        default="openrouter",
        choices=["mock", "openrouter", "openai"],
        help="Model provider for G-Eval judging",
    )
    parser.add_argument(
        "--evaluator-model",
        default="gpt-4o",
        help="Model used for G-Eval scoring",
    )
    parser.add_argument(
        "--skip-geval",
        action="store_true",
        help="Skip the G-Eval model-based evaluation pass.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print supported OpenRouter model aliases and exit.",
    )
    parser.add_argument("--sample-size", type=int, default=10, help="Samples per dataset position")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=220, help="Max output tokens")
    parser.add_argument("--evaluator-temperature", type=float, default=0.0, help="Temperature for G-Eval")
    parser.add_argument("--evaluator-max-tokens", type=int, default=80, help="Max tokens for G-Eval")
    parser.add_argument("--top-csv", default="positional_bias_concatenated_top.csv")
    parser.add_argument("--middle-csv", default="positional_bias_concatenated_middle.csv")
    parser.add_argument("--end-csv", default="positional_bias_concatenated_end.csv")
    parser.add_argument("--output-dir", default="experiment_outputs")
    return parser.parse_args()


def main():
    workspace = Path(__file__).resolve().parent
    load_env_file(workspace / ".env")
    args = parse_args()

    if args.list_models:
        print("Supported OpenRouter model aliases:")
        for alias, full_name in OPENROUTER_MODELS.items():
            print(f"- {alias} -> {full_name}")
        return

    datasets = {
        "top": (workspace / args.top_csv).resolve(),
        "middle": (workspace / args.middle_csv).resolve(),
        "end": (workspace / args.end_csv).resolve(),
    }

    client = build_model_client(args.provider)
    evaluator_client = None if args.skip_geval else build_model_client(args.evaluator_provider)
    config = RunConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=(workspace / args.output_dir).resolve(),
        evaluator_model=args.evaluator_model,
        evaluator_temperature=args.evaluator_temperature,
        evaluator_max_tokens=args.evaluator_max_tokens,
        enable_geval=not args.skip_geval,
    )

    log_path, detailed_path, summary_path, records, summary_rows = run_experiments(
        client=client,
        evaluator_client=evaluator_client,
        datasets=datasets,
        config=config,
    )

    print("Experiment run complete")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    if args.skip_geval:
        print("G-Eval: skipped")
    else:
        print(f"G-Eval provider: {args.evaluator_provider}")
        print(f"G-Eval model: {args.evaluator_model}")
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
                f"sentiment_deviation={row['avg_sentiment_deviation']}, "
                f"geval={row['avg_geval_score']}, "
                f"cosine_similarity={row['avg_negative_review_cosine_similarity']}"
            )


if __name__ == "__main__":
    main()

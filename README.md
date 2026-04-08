# NLP Project: Yelp Positional Bias Pipeline

This project builds a filtered Yelp dataset, creates positional variants of review concatenations, and runs summarization experiments with model APIs (including OpenRouter).

## Prerequisites

- Python 3.10+
- Install dependencies:

    pip install openai

- Configure environment values in .env
  - OPENROUTER_API_KEY=your_key
  - OPENROUTER_SITE_URL=https://your-site.example (optional)
  - OPENROUTER_APP_NAME=nlp_project (optional)
  - OPENAI_API_KEY=... (optional, only if using --provider openai)

## Executable Files and Commands

Run all commands from the project root folder (the folder containing these scripts).

### 1) Analyze businesses and review counts

File: analyze_yelp_reviews.py

Purpose:
- Reads business.json and review.json
- Computes per-business counts
- Reports how many businesses meet this requirement:
  - at least 10 five-star reviews
  - at least 1 negative review (1-2 stars)
- Writes business_review_stats.csv

Command:

    python analyze_yelp_reviews.py

Optional arguments:

    python analyze_yelp_reviews.py --business-file business.json --review-file review.json --output-csv business_review_stats.csv

### 2) Create positional-bias base dataset (10 five-star + 1 negative per business)

File: create_positional_bias_dataset.py

Purpose:
- Reads business_review_stats.csv
- Filters businesses using meets_10_five_star_1_negative
- Samples businesses and creates positional_bias_dataset.csv

Command (default 100 businesses):

    python create_positional_bias_dataset.py

Small test run:

    python create_positional_bias_dataset.py --business-limit 10

Optional arguments:

    python create_positional_bias_dataset.py --business-limit 100 --seed 42 --stats-file business_review_stats.csv --output-csv positional_bias_dataset.csv

### 3) Create concatenated datasets with negative review at top/middle/end

File: create_positioned_concatenations.py

Purpose:
- Reads positional_bias_dataset.csv
- Produces:
  - positional_bias_concatenated_top.csv
  - positional_bias_concatenated_middle.csv
  - positional_bias_concatenated_end.csv

Command:

    python create_positioned_concatenations.py

Optional arguments:

    python create_positioned_concatenations.py --input-csv positional_bias_dataset.csv --output-prefix positional_bias_concatenated

### 4) Run summarization experiments

File: run_experiments.py

Purpose:
- Runs strategies (baseline, cot, repetition) across top/middle/end datasets
- Saves logs and metrics to experiment_outputs

List supported OpenRouter aliases:

    python run_experiments.py --list-models

Run with OpenRouter (example: GPT-4o mini):

    python run_experiments.py --provider openrouter --model "gpt-4o-mini" --sample-size 10

Run with your other models:

    python run_experiments.py --provider openrouter --model "gemini 2.0 flash-lite" --sample-size 10
    python run_experiments.py --provider openrouter --model "gemini 2.0 flash" --sample-size 10
    python run_experiments.py --provider openrouter --model "gpt-4o mini" --sample-size 10
    python run_experiments.py --provider openrouter --model "claude 3 haiku" --sample-size 10
    python run_experiments.py --provider openrouter --model "claude 3.7 sonnet" --sample-size 10

Optional arguments:

    python run_experiments.py --provider openrouter --model "gpt-4o-mini" --sample-size 100 --temperature 0.2 --max-tokens 220 --output-dir experiment_outputs

## Recommended End-to-End Order

1. python analyze_yelp_reviews.py
2. python create_positional_bias_dataset.py --business-limit 100
3. python create_positioned_concatenations.py
4. python run_experiments.py --provider openrouter --model "gpt-4o-mini" --sample-size 10
5. Increase sample size after a successful small run

## Outputs

- business_review_stats.csv
- positional_bias_dataset.csv
- positional_bias_concatenated_top.csv
- positional_bias_concatenated_middle.csv
- positional_bias_concatenated_end.csv
- experiment_outputs/experiment_logs_*.jsonl
- experiment_outputs/experiment_detailed_*.csv
- experiment_outputs/experiment_summary_*.csv

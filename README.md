# NLP Project: Yelp Positional Bias Pipeline

This project builds a filtered Yelp dataset, creates positional variants of review concatenations, and runs summarization experiments with model APIs (including OpenRouter).

## Prerequisites

- Python 3.10+
- Install dependencies:

    pip install openai nltk

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

### 2) Create positional-bias base dataset (configurable positive/negative counts)

File: create_positional_bias_dataset.py

Purpose:
- Reads business_review_stats.csv
- Filters businesses by the requested number of five-star and negative reviews
- Samples businesses and creates positional_bias_dataset.csv
- Stores the sampled review counts in the output CSV

Command (default 100 businesses, 10 positive + 1 negative):

    python create_positional_bias_dataset.py

Small test run:

    python create_positional_bias_dataset.py --business-limit 10 --positive-count 5 --negative-count 1

Optional arguments:

    python create_positional_bias_dataset.py --business-limit 100 --positive-count 10 --negative-count 1 --seed 42 --stats-file business_review_stats.csv --output-csv positional_bias_dataset.csv
block at top/middle/end

File: create_positioned_concatenations.py

Purpose:
- Reads positional_bias_dataset.csv
- Produces:
  - positional_bias_concatenated_top.csv
  - positional_bias_concatenated_middle.csv
  - positional_bias_concatenated_end.csv
- Preserves the sampled positive/negative counts for each businesscsv
  - positional_bias_concatenated_end.csv

Command:

    python create_positioned_concatenations.py

Optional arguments:

    python create_positioned_concatenations.py --input-csv positional_bias_dataset.csv --output-prefix positional_bias_concatenated

### 4) Run summarization experiments

File: run_experiments.py

Purpose:
- Rcores each generated summary with VADER against the expected aggregate sentiment
- Computes G-Eval with a separate judge model
- Computes cosine similarity between the negative-review block and the summary
- Saves logs and metrics to experiment_outputs

List supported OpenRouter aliases:

    python run_experiments.py --list-models

Run with OpenRouter (example: GPT-4o mini):

    python run_experiments.py --provider openrouter --model "gpt-4o-mini" --sample-size 10

Run G-Eval with GPT-4o:

    python run_experiments.py --provider openrouter --model "gpt-4o-mini" --evaluator-provider openrouter --evaluator-model "gpt-4o" --sample-size 10

Optional arguments:

    python run_experiments.py --provider openrouter --model "gpt-4o-mini" --evaluator-provider openrouter --evaluator-model "gpt-4o" --sample-size 100 --temperature 0.2 --max-tokens 220 --evaluator-temperature 0.0 --evaluator-max-tokens 80 --output-dir experiment_outputs

Use `--skip-geval` if you want to run only the VADER and cosine-similarity evaluations.
    python run_experiments.py --provider openrouter --model "gpt-4o-mini" --sample-size 100 --temperature 0.2 --max-tokens 220 --output-dir experiment_outputs

## Recommended End-to-End Order
 --positive-count 10 --negative-count 1
3. python create_positioned_concatenations.py
4. python run_experiments.py --provider openrouter --model "gpt-4o-mini" --evaluator-provider openrouter --evaluator-model "gpt-4o" --sample-size 10
5. Increase sample size or adjust positive/negative countsoned_concatenations.py
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

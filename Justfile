#!/usr/bin/env just --justfile

# Default task to run if no other is specified.
default:
	@just -l

# Check if uv is installed
check-uv:
	@command -v uv >/dev/null 2>&1 || (echo "uv is missing. Install uv and run the command again." && exit 1)

# Check if raw data exists
check-data: check-uv
	@test -d data_raw || (echo "Missing data_raw/ directory." && exit 1)
	@find data_raw -name "*.csv" | grep -q . || (echo "No CSV files found in data_raw/." && exit 1)

# Install dependencies
install: check-uv
	uv sync

# --- Linting and Formatting ---
lint: check-uv
	uv run ruff check scripts

format: check-uv
	uv run ruff format scripts

check: lint
	uv run ruff format --check scripts


# Run the paper baseline variant.
paper: check-data
	uv run python scripts/preprocess_data_paper_baseline.py
	uv run python scripts/train_random_forest_paper_baseline.py

# Run the controlled variant with deduplication.
controlled: check-data
	uv run python scripts/preprocess_data_controlled.py
	uv run python scripts/train_random_forest_controlled.py

# Run Brute Force detection pipeline
bruteforce: check-data
	uv run python scripts/preprocess_data_bruteforce.py
	uv run python scripts/train_random_forest_bruteforce.py

# Run Port Scan detection pipeline
portscan: check-data
	uv run python scripts/preprocess_data_portscan.py
	uv run python scripts/train_random_forest_portscan.py

# Run Web Attacks pipeline
web_attacks: check-data
	uv run python scripts/preprocess_data_web_attacks.py
	uv run python scripts/train_random_forest_web_attacks.py

# Prepare the final CICIDS2017 binary dataset.
preprocess-cicids2017: check-data
	uv run python scripts/01_preprocess_cicids2017_binary.py

# Train the final Random Forest binary IDS model.
train-binary: check-data
	uv run python scripts/02_train_random_forest_binary.py

# Prepare CSE-CIC-IDS2018 as an external-only test set.
preprocess-cse2018: check-data
	uv run python scripts/03_preprocess_cse_cic_ids2018_external.py

# Evaluate the final model on internal and external test sets.
evaluate-binary: check-data
	uv run python scripts/04_evaluate_binary_model.py

# Fast smoke-test evaluation on a limited number of rows per dataset.
evaluate-binary-quick: check-data
	uv run python scripts/04_evaluate_binary_model.py --max-rows 200000 --skip-predictions --name-suffix _quick

# Run SHAP analysis for the final binary model.
shap-binary: check-data
	uv run python scripts/05_analyze_shap_binary.py

# Fast smoke-test SHAP analysis on a small sample.
shap-binary-quick: check-data
	uv run python scripts/05_analyze_shap_binary.py --max-rows 200000 --samples-per-family 300 --name-suffix _quick

# Run the final model pipeline without SHAP.
main: preprocess-cicids2017 train-binary preprocess-cse2018 evaluate-binary

# Run the final model pipeline including SHAP.
main-with-shap: main shap-binary

# Run the final model pipeline with quick evaluation and quick SHAP.
main-quick: preprocess-cicids2017 train-binary preprocess-cse2018 evaluate-binary-quick shap-binary-quick

# Backward-compatible aliases for the final model pipeline.
full_dataset: main
generalization: main

# Run legacy exploratory variants.
legacy-all: bruteforce web_attacks portscan paper controlled

# Run the default final pipeline.
all: main

# Summarize results from all variants
summarize-results: check-data
	uv run python scripts/summarize_variant_results.py

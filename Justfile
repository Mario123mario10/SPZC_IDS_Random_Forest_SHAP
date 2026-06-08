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
	uv run python scripts/reproduction/preprocess_data_paper_baseline.py
	uv run python scripts/reproduction/train_random_forest_paper_baseline.py

# Run the controlled variant with deduplication.
controlled: check-data
	uv run python scripts/reproduction/preprocess_data_controlled.py
	uv run python scripts/reproduction/train_random_forest_controlled.py

# Run Brute Force detection pipeline
bruteforce: check-data
	PYTHONPATH=scripts/reproduction uv run python scripts/legacy/attack_specific/preprocess_data_bruteforce.py
	PYTHONPATH=scripts/reproduction uv run python scripts/legacy/attack_specific/train_random_forest_bruteforce.py

# Run Port Scan detection pipeline
portscan: check-data
	PYTHONPATH=scripts/reproduction uv run python scripts/legacy/attack_specific/preprocess_data_portscan.py
	PYTHONPATH=scripts/reproduction uv run python scripts/legacy/attack_specific/train_random_forest_portscan.py

# Run Web Attacks pipeline
web_attacks: check-data
	PYTHONPATH=scripts/reproduction uv run python scripts/legacy/attack_specific/preprocess_data_web_attacks.py
	PYTHONPATH=scripts/reproduction uv run python scripts/legacy/attack_specific/train_random_forest_web_attacks.py

# Prepare the final CICIDS2017 binary dataset.
preprocess-cicids2017: check-data
	uv run python scripts/main/01_preprocess_cicids2017_binary.py

# Train the final Random Forest binary IDS model.
train-binary: check-data
	uv run python scripts/main/02_train_random_forest_binary.py

# Prepare CSE-CIC-IDS2018 as an external-only test set.
preprocess-cse2018: check-data
	uv run python scripts/main/03_preprocess_cse_cic_ids2018_external.py

# Evaluate the final model on internal and external test sets.
evaluate-binary: check-data
	uv run python scripts/main/04_evaluate_binary_model.py

# Fast smoke-test evaluation on a limited number of rows per dataset.
evaluate-binary-quick: check-data
	uv run python scripts/main/04_evaluate_binary_model.py --max-rows 200000 --skip-predictions --name-suffix _quick

# Run SHAP analysis for the final binary model.
shap-binary: check-data
	uv run python scripts/main/05_analyze_shap_binary.py

# Fast smoke-test SHAP analysis on a small sample.
shap-binary-quick: check-data
	uv run python scripts/main/05_analyze_shap_binary.py --max-rows 200000 --samples-per-family 300 --name-suffix _quick

# Analyze precision/recall trade-offs for different decision thresholds.
threshold-sweep:
	uv run python scripts/main/06_threshold_sweep.py

# Fast smoke-test threshold analysis on existing prediction files.
threshold-sweep-quick:
	uv run python scripts/main/06_threshold_sweep.py --max-rows 200000

# Train and evaluate the ablation variant without dst_port and fwd_header_len.
ablation-no-port-header: check-data
	uv run python scripts/main/07_ablation_no_dst_port_no_fwd_header_len.py

# Fast smoke-test of the ablation variant evaluation.
ablation-no-port-header-quick: check-data
	uv run python scripts/main/07_ablation_no_dst_port_no_fwd_header_len.py --max-rows 200000

# Run the final model pipeline without SHAP.
main: preprocess-cicids2017 train-binary preprocess-cse2018 evaluate-binary

# Run the final model pipeline including SHAP.
main-with-shap: main shap-binary

# Run the final model pipeline with quick evaluation and quick SHAP.
main-quick: preprocess-cicids2017 train-binary preprocess-cse2018 evaluate-binary-quick shap-binary-quick

# Run legacy exploratory variants.
legacy-all: bruteforce web_attacks portscan paper controlled

# Run the default final pipeline.
all: main

# Summarize results from all variants
summarize-results: check-data
	uv run python scripts/reproduction/summarize_variant_results.py

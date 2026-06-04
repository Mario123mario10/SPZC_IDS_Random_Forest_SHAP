#!/usr/bin/env just --justfile

# Default task to run if no other is specified.
default: all

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

# Run the full dataset variant with sampling
full_dataset: check-data
	uv run python scripts/preprocess_data_full_dataset.py
	uv run python scripts/train_random_forest_full_dataset.py

# Run the generalization test (train on 2017, test on 2018)
generalization: check-data
	uv run python scripts/preprocess_data_generalization_test.py
	uv run python scripts/train_and_evaluate_generalization.py

# Run the main variants.
all: bruteforce web_attacks full_dataset generalization portscan paper controlled

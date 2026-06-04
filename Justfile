# List available commands
@default:
	just -l

# Check if uv is installed
check-uv:
	@command -v uv >/dev/null 2>&1 || (echo "uv is missing. Install uv and run the command again." && exit 1)

# Check if Python is available through uv
check-python: check-uv
	@uv run python --version

# Check Python code with Ruff
lint: check-uv
	uv run ruff check scripts

# Auto-fix safe Ruff issues
lint-fix: check-uv
	uv run ruff check --fix scripts

# Format Python code with Ruff
format: check-uv
	uv run ruff format scripts

# Check formatting without changing files
format-check: check-uv
	uv run ruff format --check scripts

# Run all static checks
check: lint format-check

# Check if raw data exists
check-data:
	@test -d data_raw || (echo "Missing data_raw/ directory." && exit 1)
	@find data_raw -name "*.csv" | grep -q . || (echo "No CSV files found in data_raw/." && exit 1)

# Generic check-data for rules that might use either or both, or just to ensure data_raw exists
check-data: check-uv
	@test -d data_raw || (echo "Missing data_raw/ directory." && exit 1)

# Install dependencies
install: check-uv check-python
	uv sync

# Preprocess CICIDS2017 data in the paper-like variant
preprocess-paper: check-uv check-data
	uv run python scripts/preprocess_data_paper_baseline.py

# Train Random Forest in the paper-like variant
train-paper: check-uv
	uv run python scripts/train_random_forest_paper_baseline.py

# Preprocess CICIDS2017 data in the controlled variant
preprocess-controlled: check-uv check-data
	uv run python scripts/preprocess_data_controlled.py

# Train Random Forest in the controlled variant
train-controlled: check-uv
	uv run python scripts/train_random_forest_controlled.py

# Backward-compatible alias for the paper-like preprocessing
preprocess: preprocess-paper

# Run paper-like baseline pipeline
paper: preprocess-paper train-paper

# Run controlled pipeline
controlled: preprocess-controlled train-controlled

# Preprocess data for Brute Force detection (can be dataset-agnostic or specific)
preprocess-bruteforce: check-uv check-data
	uv run python scripts/preprocess_data_bruteforce.py

# Train Random Forest for Brute Force detection
train-bruteforce: check-uv
	uv run python scripts/train_random_forest_bruteforce.py

# Preprocess data for Port Scan detection (can be dataset-agnostic or specific)
preprocess-portscan: check-uv check-data
	uv run python scripts/preprocess_data_portscan.py

# Train Random Forest for Port Scan detection
train-portscan: check-uv
	uv run python scripts/train_random_forest_portscan.py

# Run default pipeline
# The 'all' rule should probably run all available pipelines, or a sensible default set.
# For now, let's make it run the paper-like baselines for both datasets and brute force.
all: paper bruteforce portscan

# Run Brute Force detection pipeline
bruteforce: preprocess-bruteforce train-bruteforce

# Run Port Scan detection pipeline
portscan: preprocess-portscan train-portscan


# List available commands
@default:
	just -l

# Check if uv is installed
check-uv:
	@command -v uv >/dev/null 2>&1 || (echo "Brakuje uv. Zainstaluj uv i uruchom ponownie komendę." && exit 1)

# Check if Python is available through uv
check-python: check-uv
	@uv run python --version

# Check if raw data exists
check-data:
	@test -d data_raw || (echo "Brakuje katalogu data_raw/." && exit 1)
	@find data_raw -name "*.csv" | grep -q . || (echo "Brakuje plików CSV w data_raw/." && exit 1)

# Install dependencies
install: check-uv check-python
	uv sync

# Preprocess CICIDS2017 data
preprocess: check-uv check-data
	uv run python scripts/preprocess_data.py

# Run full pipeline
all: preprocess
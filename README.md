# SPZC IDS Random Forest + SHAP

This project implements and analyzes an intrusion detection system based on the
CICIDS2017 dataset. The pipeline prepares flow data, trains Random Forest
models, evaluates detection quality, and explains model decisions with SHAP.

## Requirements

The project assumes the following tools:

- Python >= 3.10
- uv
- just

The `uv` and `just` tools should be installed globally and available from the
terminal.

## Data Layout

The full CICIDS2017 dataset is not stored in the repository because of file
size. Download the CSV files from the Canadian Institute for Cybersecurity and
place them in:

```text
data_raw/
```

After preprocessing, generated files are written to:

```text
data_processed/
models/
```

Variant-specific outputs are stored separately:

```text
data_processed/paper_baseline/
data_processed/controlled/
models/paper_baseline/
models/controlled/
```

Directory structure:

```text
project/
|-- data_raw/
|-- data_processed/
|-- models/
|-- scripts/
|-- pyproject.toml
|-- uv.lock
|-- Justfile
`-- README.md
```

## Installation

Recommended dependency installation:

```bash
uv sync
```

Alternatively, use:

```bash
just install
```

## Running The Pipeline

Run the paper-like baseline:

```bash
just paper
```

Run the controlled variant:

```bash
just controlled
```

Run the default pipeline:

```bash
just all
```

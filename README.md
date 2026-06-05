# SPZC IDS Random Forest + SHAP

This project implements and analyzes an intrusion detection system based on
Random Forest and SHAP. The final pipeline trains one binary model on
CICIDS2017 (`Benign` vs `Attack`) and evaluates the same model on an external
CSE-CIC-IDS2018 test set without retraining.

## Requirements

The project assumes the following tools:

- Python >= 3.10
- uv
- just

The `uv` and `just` tools should be installed globally and available from the
terminal.

## Data Layout

The raw datasets are not stored in the repository because of file size. Place
the CSV files in separate directories:
```text
data_raw/
  CIC_IDS2017/
  CSE_CIC_IDS2018/
```

After preprocessing, generated files are written to:

```text
data_processed/
models/
reports/
```

The final binary pipeline writes:

```text
data_processed/cicids2017_binary/
data_processed/cse_cic_ids2018_external/
models/binary_ids/
reports/tables/
reports/figures/
```

Legacy exploratory variants are still stored separately:

```text
data_processed/paper_baseline/
data_processed/controlled/
data_processed/portscan/
data_processed/bruteforce/
data_processed/web_attacks/
models/paper_baseline/
models/controlled/
models/portscan/
models/bruteforce/
models/web_attacks/
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

Show available commands:

```bash
just
```

Run the final binary IDS pipeline without SHAP:

```bash
just main
```

Run a faster smoke test after preprocessing/training:

```bash
just evaluate-binary-quick
just shap-binary-quick
```

Run the full preprocessing/training sequence with quick evaluation and quick
SHAP:

```bash
just main-quick
```

Run the final binary IDS pipeline with SHAP:

```bash
just main-with-shap
```

The final pipeline executes:

1. `scripts/01_preprocess_cicids2017_binary.py`
2. `scripts/02_train_random_forest_binary.py`
3. `scripts/03_preprocess_cse_cic_ids2018_external.py`
4. `scripts/04_evaluate_binary_model.py`
5. `scripts/05_analyze_shap_binary.py` only in `main-with-shap`

The model is trained only on CICIDS2017. CSE-CIC-IDS2018 is transformed with
the CICIDS2017 scaler and used only for external validation.

The external CSE-CIC-IDS2018 test set can contain millions of rows. Use
`evaluate-binary-quick` or `shap-binary-quick` for fast checks, and
`evaluate-binary` / `shap-binary` for the full final results.

Run the paper-like baseline:

```bash
just paper
```

Run the controlled variant:

```bash
just controlled
```

Run all legacy exploratory variants:

```bash
just legacy-all
```

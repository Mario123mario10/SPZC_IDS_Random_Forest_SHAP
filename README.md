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

## Results

### Variant Comparison

| variant            | accuracy | precision | recall | f1\_score | test samples |
| :----------------- | -------: | --------: | -----: | --------: | -----------: |
| paper\_baseline    |   0.9999 |    1.0000 | 0.9995 |    0.9997 |      131 502 |
| controlled         |   0.9999 |    0.9999 | 0.9996 |    0.9997 |      145 135 |
| main\_internal\_2017 | 0.9989 |    0.9963 | 0.9993 |    0.9978 |      107 516 |
| main\_external\_2018 | 0.7235 |    0.9696 | 0.0325 |    0.0629 |    8 057 736 |

Using a set of CICIDS2017 data, the paper's baseline and controlled variants score almost 100%.
The main binary model works good inside (2017 test split), however it works bad on the external CSE-CIC-IDS2018 set. Its high precision and near-zero recall show that almost all 2018 traffic is classified as benign, indicating a significant distribution shift between the two datasets.

---

### Paper Baseline

Dataset: Monday benign traffic + Friday DDoS traffic (CICIDS2017).
No deduplication step; class imbalance left as-is (≈ 4:1 benign:attack ratio).

| metric    |   value |
| :-------- | ------: |
| Accuracy  |  0.9999 |
| Precision |  1.0000 |
| Recall    |  0.9995 |
| F1-score  |  0.9997 |

Confusion matrix (test set - 131 502 samples):

```
              Predicted Benign   Predicted Attack
Actual Benign        105 896                  1
Actual Attack             13             25 592
```

#### Top 10 features by SHAP importance (paper baseline)

| rank | feature                    | mean \|SHAP\| |
| ---: | :------------------------- | -----------: |
|    1 | Packet Length Variance     |      0.04503 |
|    2 | Packet Length Std          |      0.04040 |
|    3 | Max Packet Length          |      0.03923 |
|    4 | Bwd Packet Length Max      |      0.02844 |
|    5 | Average Packet Size        |      0.02663 |
|    6 | Avg Bwd Segment Size       |      0.02641 |
|    7 | Total Length of Bwd Packets |     0.02300 |
|    8 | Fwd Packet Length Max      |      0.02027 |
|    9 | Subflow Bwd Packets        |      0.01982 |
|   10 | Destination Port           |      0.01190 |

The importance of features is determined by packet-length data, which is consistent with DDoS behavior generating constant, high-volume packets that are distinct from typical online browsing.

---

### Controlled Variant

Dataset: Monday benign + Friday benign + Friday DDoS (CICIDS2017).
Adds Friday benign traffic to reduce the temporal confound present in the paper baseline.
Deduplication removed 29 518 rows (3.91 %).

| metric    |         mean |          std |
| :-------- | -----------: | -----------: |
| Accuracy  |    0.9999170 | 4.87 × 10⁻⁶  |
| Precision |    0.9999220 | 2.76 × 10⁻⁵  |
| Recall    |    0.9996090 | 0.00          |
| F1-score  |    0.9997660 | 1.38 × 10⁻⁵  |

Results are averaged over 5 random seeds (42, 100, 2023, 777, 1337).

No Destination Port experiment (5-seed average):

| metric    |         mean |          std |
| :-------- | -----------: | -----------: |
| Accuracy  |    0.9998660 | 1.73 × 10⁻⁵  |
| Precision |    0.9996480 | 7.81 × 10⁻⁵  |
| Recall    |    0.9995940 | 2.14 × 10⁻⁵  |
| F1-score  |    0.9996210 | 4.90 × 10⁻⁵  |

Removing Destination Port causes a negligible drop in F1 (−0.0001), confirming that
packet-length and flow-byte features alone are sufficient for DDoS detection in this dataset.

#### Top 10 features by SHAP importance (controlled, seed 42)

| rank | feature                    | mean \|SHAP\| |
| ---: | :------------------------- | -----------: |
|    1 | Bwd Packet Length Max      |      0.04830 |
|    2 | Avg Bwd Segment Size       |      0.03716 |
|    3 | Packet Length Variance     |      0.03229 |
|    4 | Max Packet Length          |      0.02993 |
|    5 | Fwd Packet Length Max      |      0.02267 |
|    6 | Avg Fwd Segment Size       |      0.01998 |
|    7 | Bwd Packet Length Mean     |      0.01990 |
|    8 | Total Length of Bwd Packets |     0.01985 |
|    9 | Packet Length Std          |      0.01975 |
|   10 | Bwd Packet Length Std      |      0.01889 |

Because Friday benign traffic produces various backward-flow patterns that the model learns to evaluate against DDoS, controlled variant slightly changes top importance toward backward-direction attributes (Bwd Packet Length Max rises to rank 1).
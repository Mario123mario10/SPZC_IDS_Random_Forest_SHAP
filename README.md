# SPZC IDS Random Forest + SHAP

This project implements and analyzes an intrusion detection system based on
Random Forest and SHAP interpretability. The starting point was a reconstruction
of the approach from the paper *Intrusion Detection using Random Forest and SHAP*
on CICIDS2017. The project then extends the original setup with a more
controlled variant and a final binary model evaluated on the independent
CSE-CIC-IDS2018 dataset.

The main pipeline trains a binary `Benign` vs `Attack` model on CICIDS2017 and
evaluates the same model on CSE-CIC-IDS2018 without retraining.

## Requirements

The project is intended to run in a Linux/WSL environment.

Required tools:

- Python >= 3.10
- `uv`
- `just`

Install dependencies:

```bash
uv sync
```

Alternatively:

```bash
just install
```

## Data

Raw datasets are not stored in the repository because of their size. Place the
CSV files under `data_raw/`:

```text
data_raw/
  CIC_IDS2017/
    Monday-WorkingHours.pcap_ISCX.csv
    Tuesday-WorkingHours.pcap_ISCX.csv
    Wednesday-workingHours.pcap_ISCX.csv
    Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    Friday-WorkingHours-Morning.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

  CSE_CIC_IDS2018/
    FTP-BruteForce.csv
    SSH-Bruteforce.csv
    Brute Force -Web.csv
    Brute Force -XSS.csv
    DoS attacks-GoldenEye.csv
    DoS attacks-Hulk.csv
    DoS attacks-SlowHTTPTest.csv
    DoS attacks-Slowloris.csv
    DDoS attacks-LOIC-HTTP.csv
    DDOS attack-HOIC.csv
    DDOS attack-LOIC-UDP.csv
```

Download sources:

- CICIDS2017: <https://cicresearch.ca/CICDataset/CIC-IDS-2017/>
- CSE-CIC-IDS2018: <https://data.mendeley.com/datasets/29hdbdzx2r/1>

Additional `README.md` files in `data_raw/`, `data_raw/CIC_IDS2017/`, and
`data_raw/CSE_CIC_IDS2018/` describe the expected data layout.

Only placeholder `README.md` files from `data_raw/` are tracked by git. The raw
CSV files should stay local and are ignored by `.gitignore`.

## Repository Structure

```text
.
|-- data_raw/          # raw datasets; not versioned
|-- data_processed/    # generated datasets after preprocessing
|-- models/            # saved models, scalers, and metadata
|-- reports/
|   |-- figures/       # result plots and SHAP figures
|   `-- tables/        # metrics, confusion matrices, predictions
|-- scripts/
|   |-- main/          # final main-model pipeline
|   |-- reproduction/  # paper baseline reconstruction and controlled variant
|   `-- legacy/        # older supporting experiments
|-- Justfile
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

## Main Commands

List available tasks:

```bash
just
```

Run the main pipeline without full SHAP analysis:

```bash
just main
```

Run the main pipeline with SHAP analysis:

```bash
just main-with-shap
```

Run a faster smoke-test version:

```bash
just main-quick
```

Analyze decision thresholds from saved prediction files:

```bash
just threshold-sweep
```

Run the ablation variant without `dst_port` and `fwd_header_len`:

```bash
just ablation-no-port-header
```

Run the paper-like reconstruction:

```bash
just paper
```

Run the controlled variant:

```bash
just controlled
```

## Main Pipeline

The main model pipeline is implemented in `scripts/main/`:

1. `01_preprocess_cicids2017_binary.py` - prepares the binary CICIDS2017
   dataset (`Benign` vs `Attack`).
2. `02_train_random_forest_binary.py` - trains the Random Forest model.
3. `03_preprocess_cse_cic_ids2018_external.py` - prepares CSE-CIC-IDS2018 as
   an external test set.
4. `04_evaluate_binary_model.py` - evaluates the model on internal and
   external test sets.
5. `05_analyze_shap_binary.py` - runs SHAP analysis for the main model.
6. `06_threshold_sweep.py` - analyzes different decision thresholds.
7. `07_ablation_no_dst_port_no_fwd_header_len.py` - trains and evaluates the
   variant without `dst_port` and `fwd_header_len`.

The model is trained only on CICIDS2017. CSE-CIC-IDS2018 is used only as an
external generalization test set.

## Results

### Variant Comparison

| variant            | accuracy | precision | recall | f1_score | test samples |
| :----------------- | -------: | --------: | -----: | -------: | -----------: |
| Paper Baseline     |   0.9999 |    1.0000 | 0.9995 |   0.9997 |      131 502 |
| Controlled         |   0.9999 |    0.9999 | 0.9996 |   0.9998 |      145 135 |
| Main internal 2017 |   0.9990 |    0.9970 | 0.9991 |   0.9981 |      107 516 |
| Main external 2018 |   0.7249 |    0.9755 | 0.0372 |   0.0717 |    8 057 736 |

The `Paper Baseline` and `Controlled` variants achieve almost perfect results
on CICIDS2017, which suggests that the DDoS vs benign task is very easy for
Random Forest in this configuration. The main model also performs very well on
the internal CICIDS2017 split, but generalizes poorly to CSE-CIC-IDS2018. At the
default decision threshold, the model is very conservative: it keeps high
precision but detects only a small fraction of attacks.

### Decision Thresholds on CSE-CIC-IDS2018

| threshold | precision | recall | f1_score | specificity | false positives | false negatives |
| --------: | --------: | -----: | -------: | ----------: | --------------: | --------------: |
|      0.50 |    0.9755 | 0.0372 |   0.0717 |      0.9996 |           2 147 |       2 214 439 |
|      0.30 |    0.9732 | 0.1385 |   0.2426 |      0.9985 |           8 776 |       1 981 363 |
|      0.20 |    0.9736 | 0.4070 |   0.5740 |      0.9956 |          25 421 |       1 363 985 |
|      0.10 |    0.7894 | 0.6686 |   0.7240 |      0.9287 |         410 300 |         762 261 |
|      0.05 |    0.6791 | 0.7876 |   0.7293 |      0.8513 |         856 055 |         488 467 |
|      0.01 |    0.4639 | 0.9914 |   0.6320 |      0.5422 |       2 635 635 |          19 777 |

Lowering the decision threshold significantly improves external recall, but it
also increases the number of false alarms. Among the tested values, threshold
`0.05` gives the best F1-score, while `0.10` is a more conservative compromise.

### Ablation: No `dst_port`, No `fwd_header_len`

| model / threshold | precision | recall | f1_score | specificity | false positives | false negatives |
| :---------------- | --------: | -----: | -------: | ----------: | --------------: | --------------: |
| original, 0.50    |    0.9755 | 0.0372 |   0.0717 |      0.9996 |           2 147 |       2 214 439 |
| ablation, 0.50    |    0.9639 | 0.0892 |   0.1633 |      0.9987 |           7 687 |       2 094 903 |
| original, 0.05    |    0.6791 | 0.7876 |   0.7293 |      0.8513 |         856 055 |         488 467 |
| ablation, 0.05    |    0.5409 | 0.6058 |   0.5715 |      0.7946 |       1 182 856 |         906 564 |

Removing `dst_port` and `fwd_header_len` improves recall at the default `0.50`
threshold, but it does not outperform the original model with a lower decision
threshold. This indicates that weak generalization is not caused only by these
two individual features, but by a broader distribution shift between CICIDS2017
and CSE-CIC-IDS2018.

## Important Output Files

Main-model results are written to:

```text
reports/tables/
  internal_2017_metrics.csv
  external_2018_metrics.csv
  internal_2017_confusion_matrix.csv
  external_2018_confusion_matrix.csv
  external_2018_recall_by_attack_family.csv
  external_2018_threshold_sweep.csv
  no_dst_port_no_fwd_header_len_external_2018_metrics.csv
  no_dst_port_no_fwd_header_len_external_2018_threshold_sweep.csv

reports/figures/
  internal_2017_shap_bar_attack.png
  external_2018_shap_bar_attack.png
  external_2018_threshold_sweep.png
  no_dst_port_no_fwd_header_len_external_2018_threshold_sweep.png
```

Paper-baseline and controlled-variant outputs are stored in:

```text
data_processed/paper_baseline/
data_processed/controlled/
models/paper_baseline/
models/controlled/
```

## Practical Conclusions

- Very high performance on CICIDS2017 does not guarantee good generalization to
  CSE-CIC-IDS2018.
- SHAP is useful for identifying which features drive model decisions, but
  interpretability alone does not solve dataset distribution shift.
- In IDS settings, the decision threshold should be selected deliberately,
  because missed attacks and false alarms have different costs.
- Removing individual dataset-specific features can help, but in this project
  threshold tuning had a stronger practical effect.

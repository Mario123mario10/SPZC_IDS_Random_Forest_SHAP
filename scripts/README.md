# scripts

The scripts are grouped by purpose.

```text
scripts/
  main/          final binary IDS pipeline
  reproduction/  paper-like baseline and controlled variant
  legacy/        older attack-specific and diagnostic experiments
```

Use `just` from the repository root instead of calling most scripts directly.
The main commands are:

```bash
just main
just main-with-shap
just threshold-sweep
just ablation-no-port-header
just paper
just controlled
```

`just threshold-sweep` uses existing `reports/tables/*_predictions.csv` files and checks how
lowering the attack decision threshold changes precision, recall, F1, and false positives.

`just ablation-no-port-header` trains and evaluates a separate main-model variant without
`dst_port` and `fwd_header_len`.

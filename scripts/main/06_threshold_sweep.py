from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from binary_pipeline_common import REPORTS_FIGURE_DIR, REPORTS_TABLE_DIR, ensure_dirs

CHUNK_SIZE = 500_000
DEFAULT_THRESHOLDS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.01)
DATASET_FILES = {
    "internal": ("internal_2017", REPORTS_TABLE_DIR / "internal_2017_predictions.csv"),
    "external": ("external_2018", REPORTS_TABLE_DIR / "external_2018_predictions.csv"),
}


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def parse_thresholds(raw_thresholds: str) -> list[float]:
    thresholds = [float(value.strip()) for value in raw_thresholds.split(",") if value.strip()]
    invalid = [threshold for threshold in thresholds if threshold < 0.0 or threshold > 1.0]
    if invalid:
        raise ValueError(f"Thresholds must be in [0, 1], got: {invalid}")
    return sorted(set(thresholds), reverse=True)


def empty_family_stats() -> dict[str, int]:
    return {
        "samples": 0,
        "correct": 0,
        "true_benign": 0,
        "true_attack": 0,
        "false_negatives": 0,
        "false_positives": 0,
    }


def metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, float | int]:
    accuracy = safe_divide(tp + tn, tn + fp + fn + tp)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1_score = safe_divide(2 * precision * recall, precision + recall)
    specificity = safe_divide(tn, tn + fp)
    return {
        "samples": tn + fp + fn + tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def update_counts(
    counts: dict[float, dict[str, int]],
    family_counts: dict[float, dict[str, dict[str, int]]],
    chunk: pd.DataFrame,
    thresholds: list[float],
) -> None:
    y_true = chunk["y_true"].to_numpy(dtype=np.int8)
    probabilities = chunk["attack_probability"].to_numpy(dtype=np.float64)
    families = chunk["attack_family"].astype(str).to_numpy()

    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(np.int8)
        counts[threshold]["tn"] += int(((y_true == 0) & (y_pred == 0)).sum())
        counts[threshold]["fp"] += int(((y_true == 0) & (y_pred == 1)).sum())
        counts[threshold]["fn"] += int(((y_true == 1) & (y_pred == 0)).sum())
        counts[threshold]["tp"] += int(((y_true == 1) & (y_pred == 1)).sum())

        for family in np.unique(families):
            family_mask = families == family
            family_y_true = y_true[family_mask]
            family_y_pred = y_pred[family_mask]
            stats = family_counts[threshold][family]
            stats["samples"] += int(family_mask.sum())
            stats["correct"] += int((family_y_true == family_y_pred).sum())
            stats["true_benign"] += int((family_y_true == 0).sum())
            stats["true_attack"] += int((family_y_true == 1).sum())
            stats["false_negatives"] += int(((family_y_true == 1) & (family_y_pred == 0)).sum())
            stats["false_positives"] += int(((family_y_true == 0) & (family_y_pred == 1)).sum())


def family_counts_to_frame(
    dataset_name: str, family_counts: dict[float, dict[str, dict[str, int]]]
) -> pd.DataFrame:
    rows = []
    for threshold, threshold_counts in family_counts.items():
        for attack_family, stats in sorted(threshold_counts.items()):
            expected_label = 0 if stats["true_benign"] >= stats["true_attack"] else 1
            rows.append(
                {
                    "dataset": dataset_name,
                    "threshold": threshold,
                    "attack_family": attack_family,
                    "samples": stats["samples"],
                    "expected_label": expected_label,
                    "recall_or_specificity": safe_divide(stats["correct"], stats["samples"]),
                    "false_negatives": stats["false_negatives"],
                    "false_positives": stats["false_positives"],
                }
            )
    return pd.DataFrame(rows).sort_values(["threshold", "attack_family"], ascending=[False, True])


def plot_threshold_sweep(dataset_name: str, sweep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = sweep["threshold"]

    ax.plot(x, sweep["precision"], marker="o", label="Precision")
    ax.plot(x, sweep["recall"], marker="o", label="Recall")
    ax.plot(x, sweep["f1_score"], marker="o", label="F1")
    ax.plot(x, sweep["specificity"], marker="o", label="Specificity")

    ax.set_xlabel("Decision threshold for attack probability")
    ax.set_ylabel("Score")
    ax.set_title(f"{dataset_name}: threshold sweep")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(REPORTS_FIGURE_DIR / f"{dataset_name}_threshold_sweep.png", dpi=200)
    plt.close(fig)


def analyze_predictions(
    dataset_name: str,
    predictions_file: Path,
    thresholds: list[float],
    *,
    max_rows: int | None,
) -> None:
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    counts = {threshold: {"tn": 0, "fp": 0, "fn": 0, "tp": 0} for threshold in thresholds}
    family_counts: dict[float, dict[str, dict[str, int]]] = {
        threshold: defaultdict(empty_family_stats) for threshold in thresholds
    }
    rows_seen = 0

    columns = ["attack_family", "y_true", "attack_probability"]
    for chunk in pd.read_csv(predictions_file, chunksize=CHUNK_SIZE, usecols=columns):
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()

        rows_seen += len(chunk)
        update_counts(counts, family_counts, chunk, thresholds)

    sweep_rows = []
    for threshold in thresholds:
        threshold_counts = counts[threshold]
        sweep_rows.append(
            {
                "dataset": dataset_name,
                "threshold": threshold,
                **metrics_from_counts(**threshold_counts),
            }
        )

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(REPORTS_TABLE_DIR / f"{dataset_name}_threshold_sweep.csv", index=False)
    family_counts_to_frame(dataset_name, family_counts).to_csv(
        REPORTS_TABLE_DIR / f"{dataset_name}_threshold_sweep_by_attack_family.csv",
        index=False,
    )
    plot_threshold_sweep(dataset_name, sweep)

    print(f"\n{dataset_name} threshold sweep:")
    print(sweep[["threshold", "precision", "recall", "f1_score", "specificity", "fp", "fn"]])
    print(f"Saved threshold sweep tables and plot for {dataset_name}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        choices=["all", "internal", "external"],
        default="all",
        help="Prediction files to analyze.",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(threshold) for threshold in DEFAULT_THRESHOLDS),
        help="Comma-separated decision thresholds for attack_probability.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional quick-run row limit per dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(REPORTS_TABLE_DIR, REPORTS_FIGURE_DIR)

    thresholds = parse_thresholds(args.thresholds)
    selected = ["internal", "external"] if args.datasets == "all" else [args.datasets]

    for key in selected:
        dataset_name, predictions_file = DATASET_FILES[key]
        analyze_predictions(
            dataset_name,
            predictions_file,
            thresholds,
            max_rows=args.max_rows,
        )


if __name__ == "__main__":
    main()

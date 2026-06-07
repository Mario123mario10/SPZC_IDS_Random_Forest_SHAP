from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from binary_pipeline_common import (
    BINARY_MODEL_FILE,
    CICIDS2017_TEST_FILE,
    CICIDS2017_TEST_METADATA_FILE,
    EXTERNAL_METADATA_FILE,
    EXTERNAL_TEST_FILE,
    REPORTS_TABLE_DIR,
    ensure_dirs,
    load_feature_names,
    validate_feature_columns,
)

CHUNK_SIZE = 200_000
LABEL_NAMES = {0: "Benign", 1: "Attack"}


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = cm.ravel()
    accuracy = safe_divide(tp + tn, cm.sum())
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def classification_report_from_confusion(cm: np.ndarray) -> pd.DataFrame:
    tn, fp, fn, tp = cm.ravel()
    rows = []

    class_counts = {
        0: {"tp": tn, "fp": fn, "fn": fp, "support": tn + fp},
        1: {"tp": tp, "fp": fp, "fn": fn, "support": tp + fn},
    }

    for label, counts in class_counts.items():
        precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
        f1 = safe_divide(2 * precision * recall, precision + recall)
        rows.append(
            {
                "label": LABEL_NAMES[label],
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": counts["support"],
            }
        )

    report = pd.DataFrame(rows).set_index("label")
    support_sum = report["support"].sum()
    macro = report[["precision", "recall", "f1_score"]].mean()
    weighted = (
        report[["precision", "recall", "f1_score"]]
        .multiply(report["support"], axis=0)
        .sum()
        .divide(support_sum)
    )
    accuracy = safe_divide(tn + tp, support_sum)

    report.loc["accuracy"] = {
        "precision": accuracy,
        "recall": accuracy,
        "f1_score": accuracy,
        "support": support_sum,
    }
    report.loc["macro avg"] = {
        "precision": macro["precision"],
        "recall": macro["recall"],
        "f1_score": macro["f1_score"],
        "support": support_sum,
    }
    report.loc["weighted avg"] = {
        "precision": weighted["precision"],
        "recall": weighted["recall"],
        "f1_score": weighted["f1_score"],
        "support": support_sum,
    }
    return report


def update_family_stats(
    family_stats: dict[str, dict[str, int]],
    y_true: pd.Series,
    y_pred: pd.Series,
    metadata: pd.DataFrame,
) -> None:
    analysis = metadata[["attack_family"]].copy()
    analysis["y_true"] = y_true.to_numpy(dtype=int)
    analysis["y_pred"] = y_pred.to_numpy(dtype=int)

    for attack_family, group in analysis.groupby("attack_family", sort=True):
        stats = family_stats[attack_family]
        stats["samples"] += len(group)
        stats["correct"] += int((group["y_true"] == group["y_pred"]).sum())
        stats["true_benign"] += int((group["y_true"] == 0).sum())
        stats["true_attack"] += int((group["y_true"] == 1).sum())
        stats["false_negatives"] += int(((group["y_true"] == 1) & (group["y_pred"] == 0)).sum())
        stats["false_positives"] += int(((group["y_true"] == 0) & (group["y_pred"] == 1)).sum())


def family_stats_to_frame(family_stats: dict[str, dict[str, int]]) -> pd.DataFrame:
    rows = []
    for attack_family, stats in sorted(family_stats.items()):
        expected_label = 0 if stats["true_benign"] >= stats["true_attack"] else 1
        rows.append(
            {
                "attack_family": attack_family,
                "samples": stats["samples"],
                "expected_label": expected_label,
                "recall_or_specificity": safe_divide(stats["correct"], stats["samples"]),
                "false_negatives": stats["false_negatives"],
                "false_positives": stats["false_positives"],
            }
        )
    return pd.DataFrame(rows)


def append_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
    *,
    write_header: bool,
) -> None:
    predictions.to_csv(
        output_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )


def save_evaluation_tables(
    dataset_name: str,
    cm: np.ndarray,
    family_stats: dict[str, dict[str, int]],
) -> None:
    metrics = metrics_from_confusion(cm)
    metrics_df = pd.DataFrame([{"dataset": dataset_name, "samples": int(cm.sum()), **metrics}])
    metrics_df.to_csv(REPORTS_TABLE_DIR / f"{dataset_name}_metrics.csv", index=False)

    cm_df = pd.DataFrame(
        cm,
        index=["true_benign", "true_attack"],
        columns=["pred_benign", "pred_attack"],
    )
    cm_df.to_csv(REPORTS_TABLE_DIR / f"{dataset_name}_confusion_matrix.csv")

    classification_report_from_confusion(cm).to_csv(
        REPORTS_TABLE_DIR / f"{dataset_name}_classification_report.csv"
    )
    family_stats_to_frame(family_stats).to_csv(
        REPORTS_TABLE_DIR / f"{dataset_name}_recall_by_attack_family.csv",
        index=False,
    )

    print(f"\n{dataset_name} metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Saved {dataset_name} evaluation tables in {REPORTS_TABLE_DIR}")


def evaluate_csv_pair(
    model,
    dataset_name: str,
    data_file: Path,
    metadata_file: Path,
    *,
    max_rows: int | None = None,
    write_predictions: bool = True,
) -> None:
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    feature_names = load_feature_names()
    cm_total = np.zeros((2, 2), dtype=np.int64)
    family_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "samples": 0,
            "correct": 0,
            "true_benign": 0,
            "true_attack": 0,
            "false_negatives": 0,
            "false_positives": 0,
        }
    )
    predictions_file = REPORTS_TABLE_DIR / f"{dataset_name}_predictions.csv"
    write_predictions_header = True
    rows_seen = 0

    data_chunks = pd.read_csv(data_file, chunksize=CHUNK_SIZE)
    metadata_chunks = pd.read_csv(metadata_file, chunksize=CHUNK_SIZE)

    for data_chunk, metadata_chunk in zip(data_chunks, metadata_chunks, strict=False):
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            data_chunk = data_chunk.iloc[:remaining].copy()
            metadata_chunk = metadata_chunk.iloc[:remaining].copy()

        if len(data_chunk) != len(metadata_chunk):
            raise ValueError(
                f"{dataset_name}: data chunk rows ({len(data_chunk)}) do not match "
                f"metadata chunk rows ({len(metadata_chunk)})"
            )
        rows_seen += len(data_chunk)

        X = validate_feature_columns(
            data_chunk.drop(columns=["Label"]), feature_names, dataset_name
        )
        y_true = data_chunk["Label"].astype(int)
        y_pred = pd.Series(model.predict(X), index=y_true.index).astype(int)
        y_proba = pd.Series(model.predict_proba(X)[:, 1], index=y_true.index)

        cm_total += confusion_matrix(y_true, y_pred, labels=[0, 1])
        update_family_stats(family_stats, y_true, y_pred, metadata_chunk)

        if write_predictions:
            predictions = metadata_chunk.copy()
            predictions["y_true"] = y_true.to_numpy(dtype=int)
            predictions["y_pred"] = y_pred.to_numpy(dtype=int)
            predictions["attack_probability"] = y_proba.to_numpy(dtype=float)
            append_predictions(
                predictions,
                predictions_file,
                write_header=write_predictions_header,
            )
            write_predictions_header = False

    save_evaluation_tables(dataset_name, cm_total, family_stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        choices=["all", "internal", "external"],
        default="all",
        help="Which test sets to evaluate.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit per evaluated dataset for quick smoke tests.",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Do not write row-level prediction CSV files.",
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help="Suffix appended to output table names, e.g. _quick.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(REPORTS_TABLE_DIR)

    if not BINARY_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {BINARY_MODEL_FILE}. "
            "Run scripts/02_train_random_forest_binary.py first."
        )

    print(f"Loading model: {BINARY_MODEL_FILE}")
    model = joblib.load(BINARY_MODEL_FILE)

    if args.datasets in {"all", "internal"}:
        print("Evaluating internal CICIDS2017 test set...")
        evaluate_csv_pair(
            model,
            f"internal_2017{args.name_suffix}",
            CICIDS2017_TEST_FILE,
            CICIDS2017_TEST_METADATA_FILE,
            max_rows=args.max_rows,
            write_predictions=not args.skip_predictions,
        )

    if args.datasets in {"all", "external"}:
        print("Evaluating external CSE-CIC-IDS2018 test set...")
        evaluate_csv_pair(
            model,
            f"external_2018{args.name_suffix}",
            EXTERNAL_TEST_FILE,
            EXTERNAL_METADATA_FILE,
            max_rows=args.max_rows,
            write_predictions=not args.skip_predictions,
        )

    print("\nBinary model evaluation finished.")


if __name__ == "__main__":
    main()

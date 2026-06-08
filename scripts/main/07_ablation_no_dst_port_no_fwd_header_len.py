from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from binary_pipeline_common import (
    BINARY_MODEL_DIR,
    CICIDS2017_TEST_FILE,
    CICIDS2017_TEST_METADATA_FILE,
    CICIDS2017_TRAIN_FILE,
    EXTERNAL_METADATA_FILE,
    EXTERNAL_TEST_FILE,
    RANDOM_STATE,
    REPORTS_FIGURE_DIR,
    REPORTS_TABLE_DIR,
    ensure_dirs,
    load_feature_names,
    validate_feature_columns,
    write_json,
)

VARIANT_NAME = "no_dst_port_no_fwd_header_len"
REMOVED_FEATURES = ("dst_port", "fwd_header_len")
DEFAULT_THRESHOLDS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.01)
CHUNK_SIZE = 200_000
LABEL_NAMES = {0: "Benign", 1: "Attack"}

VARIANT_MODEL_DIR = BINARY_MODEL_DIR / VARIANT_NAME
VARIANT_MODEL_FILE = VARIANT_MODEL_DIR / "random_forest_cicids2017_binary.joblib"


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def parse_thresholds(raw_thresholds: str) -> list[float]:
    thresholds = [float(value.strip()) for value in raw_thresholds.split(",") if value.strip()]
    invalid = [threshold for threshold in thresholds if threshold < 0.0 or threshold > 1.0]
    if invalid:
        raise ValueError(f"Thresholds must be in [0, 1], got: {invalid}")
    return sorted(set(thresholds), reverse=True)


def ablation_feature_names() -> list[str]:
    feature_names = load_feature_names()
    missing_removed = [feature for feature in REMOVED_FEATURES if feature not in feature_names]
    if missing_removed:
        raise ValueError(f"Cannot remove missing features: {missing_removed}")
    return [feature for feature in feature_names if feature not in REMOVED_FEATURES]


def load_training_data(feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    if not CICIDS2017_TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {CICIDS2017_TRAIN_FILE}. "
            "Run scripts/main/01_preprocess_cicids2017_binary.py first."
        )

    train_df = pd.read_csv(CICIDS2017_TRAIN_FILE)
    X_train = validate_feature_columns(train_df.drop(columns=["Label"]), feature_names, "train")
    y_train = train_df["Label"].astype(int)
    return X_train, y_train


def train_variant(feature_names: list[str]) -> None:
    ensure_dirs(VARIANT_MODEL_DIR)
    print(f"Training ablation variant: {VARIANT_NAME}")
    print(f"Removed features: {', '.join(REMOVED_FEATURES)}")
    X_train, y_train = load_training_data(feature_names)

    params = {
        "n_estimators": 50,
        "max_depth": 15,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": "balanced",
        "removed_features": list(REMOVED_FEATURES),
        "variant": VARIANT_NAME,
    }
    model_params = {
        key: value for key, value in params.items() if key not in {"removed_features", "variant"}
    }
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    joblib.dump(model, VARIANT_MODEL_FILE)
    write_json(params, VARIANT_MODEL_DIR / "model_params.json")
    write_json({"features": feature_names}, VARIANT_MODEL_DIR / "feature_names.json")

    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "gini_importance": model.feature_importances_,
        }
    ).sort_values("gini_importance", ascending=False)
    importance_df.to_csv(VARIANT_MODEL_DIR / "feature_importance_gini.csv", index=False)
    print(f"Saved ablation model: {VARIANT_MODEL_FILE}")


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = cm.ravel()
    accuracy = safe_divide(tp + tn, cm.sum())
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    specificity = safe_divide(tn, tn + fp)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
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


def update_threshold_counts(
    threshold_counts: dict[float, dict[str, int]],
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    for threshold, counts in threshold_counts.items():
        y_pred = (probabilities >= threshold).astype(np.int8)
        counts["tn"] += int(((y_true == 0) & (y_pred == 0)).sum())
        counts["fp"] += int(((y_true == 0) & (y_pred == 1)).sum())
        counts["fn"] += int(((y_true == 1) & (y_pred == 0)).sum())
        counts["tp"] += int(((y_true == 1) & (y_pred == 1)).sum())


def threshold_counts_to_frame(
    dataset_name: str, threshold_counts: dict[float, dict[str, int]]
) -> pd.DataFrame:
    rows = []
    for threshold, counts in threshold_counts.items():
        cm = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
        rows.append(
            {
                "dataset": dataset_name,
                "threshold": threshold,
                "samples": int(cm.sum()),
                **metrics_from_confusion(cm),
                **counts,
            }
        )
    return pd.DataFrame(rows).sort_values("threshold", ascending=False)


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


def evaluate_dataset(
    model,
    feature_names: list[str],
    dataset_name: str,
    data_file: Path,
    metadata_file: Path,
    thresholds: list[float],
    *,
    max_rows: int | None,
) -> None:
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
    threshold_counts = {threshold: {"tn": 0, "fp": 0, "fn": 0, "tp": 0} for threshold in thresholds}
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
        probabilities = model.predict_proba(X)[:, 1]

        cm_total += confusion_matrix(y_true, y_pred, labels=[0, 1])
        update_family_stats(family_stats, y_true, y_pred, metadata_chunk)
        update_threshold_counts(threshold_counts, y_true.to_numpy(dtype=np.int8), probabilities)

    metrics = metrics_from_confusion(cm_total)
    pd.DataFrame([{"dataset": dataset_name, "samples": int(cm_total.sum()), **metrics}]).to_csv(
        REPORTS_TABLE_DIR / f"{dataset_name}_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        cm_total,
        index=["true_benign", "true_attack"],
        columns=["pred_benign", "pred_attack"],
    ).to_csv(REPORTS_TABLE_DIR / f"{dataset_name}_confusion_matrix.csv")
    classification_report_from_confusion(cm_total).to_csv(
        REPORTS_TABLE_DIR / f"{dataset_name}_classification_report.csv"
    )
    family_stats_to_frame(family_stats).to_csv(
        REPORTS_TABLE_DIR / f"{dataset_name}_recall_by_attack_family.csv",
        index=False,
    )

    threshold_sweep = threshold_counts_to_frame(dataset_name, threshold_counts)
    threshold_sweep.to_csv(
        REPORTS_TABLE_DIR / f"{dataset_name}_threshold_sweep.csv",
        index=False,
    )
    plot_threshold_sweep(dataset_name, threshold_sweep)

    print(f"\n{dataset_name} metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(threshold_sweep[["threshold", "precision", "recall", "f1_score", "fp", "fn"]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse an existing ablation model instead of retraining.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit per evaluated dataset for quick smoke tests.",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(threshold) for threshold in DEFAULT_THRESHOLDS),
        help="Comma-separated decision thresholds for attack_probability.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(VARIANT_MODEL_DIR, REPORTS_TABLE_DIR, REPORTS_FIGURE_DIR)

    feature_names = ablation_feature_names()
    if not args.skip_train:
        train_variant(feature_names)
    elif not VARIANT_MODEL_FILE.exists():
        raise FileNotFoundError(f"Cannot reuse missing model: {VARIANT_MODEL_FILE}")

    model = joblib.load(VARIANT_MODEL_FILE)
    thresholds = parse_thresholds(args.thresholds)

    evaluate_dataset(
        model,
        feature_names,
        f"{VARIANT_NAME}_internal_2017",
        CICIDS2017_TEST_FILE,
        CICIDS2017_TEST_METADATA_FILE,
        thresholds,
        max_rows=args.max_rows,
    )
    evaluate_dataset(
        model,
        feature_names,
        f"{VARIANT_NAME}_external_2018",
        EXTERNAL_TEST_FILE,
        EXTERNAL_METADATA_FILE,
        thresholds,
        max_rows=args.max_rows,
    )

    print(f"\nAblation variant finished: {VARIANT_NAME}")


if __name__ == "__main__":
    main()

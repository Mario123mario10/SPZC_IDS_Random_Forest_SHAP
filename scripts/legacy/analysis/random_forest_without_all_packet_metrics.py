from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from variant_paths import add_variant_argument, get_variant_paths

RANDOM_STATE = 42

SIZE_VOLUME_RATE_HEADER_FEATURES = [
    # packet length
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    # byte/volume
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
    "Flow Bytes/s",
    # packet count/rate
    "Total Fwd Packets",
    "Total Backward Packets",
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "act_data_pkt_fwd",
    # header/window/segment
    "Fwd Header Length",
    "Fwd Header Length.1",
    "Bwd Header Length",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "min_seg_size_forward",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_variant_argument(parser)
    return parser.parse_args()


def load_train_test_data():
    args = parse_args()
    paths = get_variant_paths(args.variant)

    train_file = paths.train_file
    test_file = paths.test_file

    if not train_file.exists():
        raise FileNotFoundError(f"File not found: {train_file}")

    if not test_file.exists():
        raise FileNotFoundError(f"File not found: {test_file}")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    if "Label" not in train_df.columns:
        raise KeyError(f"Missing Label column in file {train_file}")

    if "Label" not in test_df.columns:
        raise KeyError(f"Missing Label column in file {test_file}")

    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"].astype(int)

    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"].astype(int)

    X_train = X_train.drop(columns=SIZE_VOLUME_RATE_HEADER_FEATURES)
    X_test = X_test.drop(columns=SIZE_VOLUME_RATE_HEADER_FEATURES)

    return X_train, X_test, y_train, y_test, paths


def main() -> None:
    print("Loading preprocessed train/test data...")
    X_train, X_test, y_train, y_test, paths = load_train_test_data()
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\nMetrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Attack"],
            zero_division=0,
        )
    )

    metrics_df = pd.DataFrame(
        [
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        ]
    )
    metrics_path = paths.processed_dir / "random_forest_without_all_packet_metrics.csv"
    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    cm_df = pd.DataFrame(
        cm,
        index=["true_benign", "true_attack"],
        columns=["pred_benign", "pred_attack"],
    )

    cm_path = paths.processed_dir / "random_forest_without_all_packet_confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    model_path = paths.model_dir / "random_forest_without_all_packet.joblib"
    joblib.dump(model, model_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()

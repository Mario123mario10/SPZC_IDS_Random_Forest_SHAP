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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from variant_paths import add_variant_argument, get_variant_paths

RANDOM_STATE = 42
TEST_SIZE = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_variant_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_variant_paths(args.variant)

    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    input_file = paths.baseline_dataset_file
    model_path = paths.model_dir / "random_forest_port_80_only.joblib"
    scaler_path = paths.processed_dir / "standard_scaler_port_80_only.joblib"
    metrics_path = paths.processed_dir / "random_forest_port_80_only_metrics.csv"
    confusion_matrix_path = paths.processed_dir / "random_forest_port_80_only_confusion_matrix.csv"

    print("Loading baseline dataset...")
    df = pd.read_csv(input_file)

    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    print("\nOriginal class distribution:")
    print(df["Label"].value_counts())

    print("\nFiltering only Destination Port == 80...")
    df_port_80 = df[df["Destination Port"] == 80].copy()

    print("\nClass distribution for port 80:")
    print(df_port_80["Label"].value_counts())
    print("\nClass distribution for port 80 normalized:")
    print(df_port_80["Label"].value_counts(normalize=True))

    if df_port_80["Label"].nunique() < 2:
        raise ValueError(
            "Only one class remains after filtering Destination Port == 80. "
            "A binary classifier cannot be trained on this subset."
        )

    y = df_port_80["Label"].apply(lambda label: 0 if label.lower() == "benign" else 1)

    X = df_port_80.drop(columns=["Label"])

    # Destination Port is constant after filtering and should not carry signal.
    X = X.drop(columns=["Destination Port"])

    non_numeric_columns = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"Non-numeric columns in X: {non_numeric_columns}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain class distribution:")
    print(y_train.value_counts())
    print("\nTest class distribution:")
    print(y_test.value_counts())

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    joblib.dump(scaler, scaler_path)

    print("\nTraining Random Forest on port 80 subset...")

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)

    print("Evaluating model...")

    y_pred = model.predict(X_test_scaled)

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
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "total_port_80_rows": len(df_port_80),
            }
        ]
    )

    metrics_df.to_csv(metrics_path, index=False)

    cm_df = pd.DataFrame(
        cm,
        index=["true_benign", "true_attack"],
        columns=["pred_benign", "pred_attack"],
    )
    cm_df.to_csv(confusion_matrix_path)

    joblib.dump(model, model_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {confusion_matrix_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

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

PROCESSED_DIR = Path("data_processed")
MODEL_DIR = Path("models")

INPUT_FILE = PROCESSED_DIR / "baseline_ddos_dataset.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_PATH = MODEL_DIR / "random_forest_port_80_only.joblib"
SCALER_PATH = PROCESSED_DIR / "standard_scaler_port_80_only.joblib"
METRICS_PATH = PROCESSED_DIR / "random_forest_port_80_only_metrics.csv"
CONFUSION_MATRIX_PATH = PROCESSED_DIR / "random_forest_port_80_only_confusion_matrix.csv"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading baseline dataset...")
    df = pd.read_csv(INPUT_FILE)

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
            "Po filtracji Destination Port == 80 została tylko jedna klasa. "
            "Nie da się trenować klasyfikatora binarnego."
        )

    y = df_port_80["Label"].apply(
        lambda label: 0 if label.lower() == "benign" else 1
    )

    X = df_port_80.drop(columns=["Label"])

    # Usuwamy Destination Port, bo po filtracji jest stały i nie powinien wnosić informacji.
    X = X.drop(columns=["Destination Port"])

    non_numeric_columns = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"Nienumeryczne kolumny w X: {non_numeric_columns}")

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

    joblib.dump(scaler, SCALER_PATH)

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

    metrics_df.to_csv(METRICS_PATH, index=False)

    cm_df = pd.DataFrame(
        cm,
        index=["true_benign", "true_attack"],
        columns=["pred_benign", "pred_attack"],
    )
    cm_df.to_csv(CONFUSION_MATRIX_PATH)

    joblib.dump(model, MODEL_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
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

from variant_paths import get_variant_paths

VARIANT = "generalization_test"
PATHS = get_variant_paths(VARIANT)
PROCESSED_DIR = PATHS.processed_dir
MODEL_DIR = PATHS.model_dir

RANDOM_STATE = 42


def load_data(file_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}. Run preprocessing first.")

    df = pd.read_csv(file_path)
    if "Label" not in df.columns:
        raise KeyError(f"Missing 'Label' column in {file_path}")

    X = df.drop(columns=["Label"])
    y = df["Label"].astype(int)
    return X, y


def evaluate_and_save(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_name: str,
) -> None:
    print(f"\n--- Evaluating on {test_name} set ---")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Metrics for {test_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"], zero_division=0))

    metrics_df = pd.DataFrame([{"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}])
    metrics_df.to_csv(PROCESSED_DIR / f"metrics_{test_name}.csv", index=False)

    cm_df = pd.DataFrame(cm, index=["true_benign", "true_attack"], columns=["pred_benign", "pred_attack"])
    cm_df.to_csv(PROCESSED_DIR / f"confusion_matrix_{test_name}.csv")

    print(f"Saved metrics and confusion matrix for {test_name} in {PROCESSED_DIR}")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data (CIC-IDS-2017)...")
    X_train, y_train = load_data(PATHS.train_file)

    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    model_path = PATHS.random_forest_model_file
    joblib.dump(model, model_path)
    print(f"Saved trained model to: {model_path}")

    print("\nLoading internal test data (CIC-IDS-2017)...")
    X_test_internal, y_test_internal = load_data(PATHS.test_file)
    evaluate_and_save(model, X_test_internal, y_test_internal, "internal_2017")

    print("\nLoading external test data (CSE-CIC-IDS-2018)...")
    X_test_external, y_test_external = load_data(PATHS.test_file_external)
    evaluate_and_save(model, X_test_external, y_test_external, "external_2018")

    print("\nGeneralization test finished.")


if __name__ == "__main__":
    main()

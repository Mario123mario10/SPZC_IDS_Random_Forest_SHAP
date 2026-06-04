from __future__ import annotations

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

from analyze_shap_baseline import generate_shap_report
from variant_paths import get_variant_paths

VARIANT = "web_attacks"
PATHS = get_variant_paths(VARIANT)
PROCESSED_DIR = PATHS.processed_dir
MODEL_DIR = PATHS.model_dir

TRAIN_FILE = PATHS.train_file
TEST_FILE = PATHS.test_file

RANDOM_STATE = 42


def load_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"File not found: {TRAIN_FILE}. Run preprocess_data_web_attacks.py first.")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"File not found: {TEST_FILE}. Run preprocess_data_web_attacks.py first.")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    if "Label" not in train_df.columns:
        raise KeyError(f"Missing Label column in file {TRAIN_FILE}")

    if "Label" not in test_df.columns:
        raise KeyError(f"Missing Label column in file {TEST_FILE}")

    if "Original_Label" not in test_df.columns:
        raise KeyError(f"Missing Original_Label column in file {TEST_FILE}. Please re-run preprocessing.")

    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"].astype(int)

    X_test = test_df.drop(columns=["Label", "Original_Label"])
    y_test = test_df["Label"].astype(int)
    y_test_original = test_df["Original_Label"]

    return X_train, X_test, y_train, y_test, y_test_original


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading preprocessed train/test data for variant '{VARIANT}'...")
    X_train, X_test, y_train, y_test, y_test_original = load_train_test_data()

    print(f"Training Random Forest for '{VARIANT}' variant...")

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
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
    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"], zero_division=0))

    metrics_df = pd.DataFrame([{"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}])
    metrics_df.to_csv(PROCESSED_DIR / "random_forest_metrics.csv", index=False)

    cm_df = pd.DataFrame(cm, index=["true_benign", "true_attack"], columns=["pred_benign", "pred_attack"])
    cm_df.to_csv(PROCESSED_DIR / "random_forest_confusion_matrix.csv")

    print("\n--- Detailed Recall per Attack Type ---")
    results_df = pd.DataFrame({"true_original": y_test_original, "predicted_binary": y_pred})
    attack_results = results_df[results_df["true_original"].str.lower() != "benign"]

    if not attack_results.empty:
        detailed_report = pd.crosstab(
            attack_results["true_original"],
            attack_results["predicted_binary"],
            rownames=["True Attack Type"],
            colnames=["Predicted (0=Benign, 1=Attack)"],
        )

        if 0 not in detailed_report.columns:
            detailed_report[0] = 0
        if 1 not in detailed_report.columns:
            detailed_report[1] = 0

        detailed_report["recall"] = detailed_report[1] / (detailed_report[0] + detailed_report[1])

        print(detailed_report)
        detailed_report_path = PROCESSED_DIR / "detailed_attack_recall.csv"
        detailed_report.to_csv(detailed_report_path)
        print(f"\nSaved detailed recall report to: {detailed_report_path}")

    model_path = PATHS.random_forest_model_file
    joblib.dump(model, model_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {PROCESSED_DIR / 'random_forest_metrics.csv'}")
    print(f"Saved confusion matrix to: {PROCESSED_DIR / 'random_forest_confusion_matrix.csv'}")

    print(f"\nGenerating SHAP report for the trained '{VARIANT}' model...")
    test_df = X_test.copy()
    test_df["Label"] = y_test.values
    generate_shap_report(model, test_df, PROCESSED_DIR / "shap_random_forest")


if __name__ == "__main__":
    main()

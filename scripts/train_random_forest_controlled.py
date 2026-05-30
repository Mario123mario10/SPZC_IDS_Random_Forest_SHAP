from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from analyze_shap_baseline import generate_shap_report
from variant_paths import get_variant_paths

VARIANT = "controlled"
PATHS = get_variant_paths(VARIANT)
PROCESSED_DIR = PATHS.processed_dir
MODEL_DIR = PATHS.model_dir
RESULTS_DIR = PROCESSED_DIR / "results"

TRAIN_FILE = PATHS.train_file
TEST_FILE = PATHS.test_file


def load_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"File not found: {TRAIN_FILE}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"File not found: {TEST_FILE}")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    if "Label" not in train_df.columns:
        raise KeyError(f"Missing Label column in file {TRAIN_FILE}")

    if "Label" not in test_df.columns:
        raise KeyError(f"Missing Label column in file {TEST_FILE}")

    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"].astype(int)

    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"].astype(int)

    return X_train, X_test, y_train, y_test


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
    }


def run_experiment(
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seeds: list[int],
    rf_params: dict[str, int],
) -> tuple[dict[str, float | str], list[RandomForestClassifier]]:
    print(f"Running experiment: {name}")
    results = []
    models = []

    for seed in seeds:
        print(f"Training with seed {seed}...")
        model = RandomForestClassifier(random_state=seed, n_jobs=-1, **rf_params)
        model.fit(X_train, y_train)
        models.append(model)

        metrics = evaluate_model(model, X_test, y_test)
        metrics["seed"] = seed
        results.append(metrics)

    df_results = pd.DataFrame(results)
    agg_metrics = {
        "experiment_name": name,
        "accuracy_mean": df_results["accuracy"].mean(),
        "accuracy_std": df_results["accuracy"].std() if len(seeds) > 1 else 0.0,
        "precision_mean": df_results["precision"].mean(),
        "precision_std": df_results["precision"].std() if len(seeds) > 1 else 0.0,
        "recall_mean": df_results["recall"].mean(),
        "recall_std": df_results["recall"].std() if len(seeds) > 1 else 0.0,
        "f1_mean": df_results["f1_score"].mean(),
        "f1_std": df_results["f1_score"].std() if len(seeds) > 1 else 0.0,
    }

    print(f"Experiment results for {name}:")
    print(f"  Accuracy:  {agg_metrics['accuracy_mean']:.4f} +/- {agg_metrics['accuracy_std']:.4f}")
    print(
        f"  Precision: {agg_metrics['precision_mean']:.4f} +/- {agg_metrics['precision_std']:.4f}"
    )
    print(f"  Recall:    {agg_metrics['recall_mean']:.4f} +/- {agg_metrics['recall_std']:.4f}")
    print(f"  F1-score:  {agg_metrics['f1_mean']:.4f} +/- {agg_metrics['f1_std']:.4f}")

    return agg_metrics, models


def find_dest_port_col(columns: pd.Index) -> str | None:
    for col in columns:
        if "destination port" in col.lower() or "dst port" in col.lower():
            return col
    return None


def save_primary_model_outputs(
    baseline_model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    model_path = PATHS.random_forest_model_file
    joblib.dump(baseline_model, model_path)
    print(f"Saved primary controlled model to: {model_path}")

    baseline_eval = evaluate_model(baseline_model, X_test, y_test)
    pd.DataFrame(
        [
            {
                "accuracy": baseline_eval["accuracy"],
                "precision": baseline_eval["precision"],
                "recall": baseline_eval["recall"],
                "f1_score": baseline_eval["f1_score"],
            }
        ]
    ).to_csv(PROCESSED_DIR / "random_forest_metrics.csv", index=False)

    cm_df = pd.DataFrame(
        baseline_eval["confusion_matrix"],
        index=["true_benign", "true_attack"],
        columns=["pred_benign", "pred_attack"],
    )
    cm_df.to_csv(PROCESSED_DIR / "random_forest_confusion_matrix.csv")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed train/test data...")
    X_train_full, X_test_full, y_train, y_test = load_train_test_data()

    rf_params = {"n_estimators": 50, "max_depth": 15}
    seeds = [42, 100, 2023, 777, 1337]

    all_experiments_results = []

    baseline_metrics, baseline_models = run_experiment(
        "Baseline",
        X_train_full,
        X_test_full,
        y_train,
        y_test,
        seeds,
        rf_params,
    )
    all_experiments_results.append(baseline_metrics)

    baseline_model = baseline_models[0]
    save_primary_model_outputs(baseline_model, X_test_full, y_test)

    print("Generating SHAP report for the controlled baseline model (seed=42)...")
    test_df = X_test_full.copy()
    test_df["Label"] = y_test.values
    generate_shap_report(
        baseline_model,
        test_df,
        PROCESSED_DIR / "shap_random_forest",
    )

    dest_port_col = find_dest_port_col(X_train_full.columns)

    if dest_port_col:
        print(f"Found Destination Port column: {dest_port_col!r}")

        X_train_no_port = X_train_full.drop(columns=[dest_port_col])
        X_test_no_port = X_test_full.drop(columns=[dest_port_col])
        no_port_metrics, _ = run_experiment(
            "No Destination Port",
            X_train_no_port,
            X_test_no_port,
            y_train,
            y_test,
            seeds,
            rf_params,
        )
        all_experiments_results.append(no_port_metrics)

        print(
            "The exact Port 80 Only experiment requires unscaled data for exact "
            "port filtering. This script runs the full feature set and the "
            "No Destination Port comparison; use random_forest_port_80_only.py "
            "for the dedicated port-80 experiment."
        )
    else:
        print("Destination Port column not found. Skipping port-specific experiments.")

    comparison_df = pd.DataFrame(all_experiments_results)
    comparison_path = RESULTS_DIR / "experiments_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved comparison table to: {comparison_path}")

    print("Final comparison:")
    print(comparison_df.to_markdown(index=False))


if __name__ == "__main__":
    main()

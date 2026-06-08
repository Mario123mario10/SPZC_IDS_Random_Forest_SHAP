from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from variant_paths import add_variant_argument, get_variant_paths

SAMPLES_PER_CLASS = 5000
RANDOM_STATE = 42
MAX_DISPLAY = 30
LOCAL_MAX_DISPLAY = 15


def get_attack_shap_values(shap_values):
    """Return SHAP values for class 1 (DDoS/Attack) across SHAP versions."""
    if isinstance(shap_values, list):
        return shap_values[1]

    if len(shap_values.shape) == 3:
        return shap_values[:, :, 1]

    return shap_values


def get_attack_expected_value(expected_value):
    if isinstance(expected_value, list):
        return expected_value[1]

    expected_array = np.asarray(expected_value)
    if expected_array.ndim == 1 and len(expected_array) > 1:
        return expected_array[1]

    return float(expected_array)


def build_balanced_sample(test_df: pd.DataFrame) -> pd.DataFrame:
    sampled_groups = []

    for _, group in test_df.groupby("Label"):
        sampled_groups.append(
            group.sample(
                n=min(len(group), SAMPLES_PER_CLASS),
                random_state=RANDOM_STATE,
            )
        )

    return pd.concat(sampled_groups, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_variant_argument(parser)
    return parser.parse_args()


def choose_local_examples(
    model,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    attack_proba = model.predict_proba(X_sample)[:, 1]
    predictions = model.predict(X_sample).astype(int)
    y_values = y_sample.to_numpy(dtype=int)

    attack_mask = (y_values == 1) & (predictions == 1)
    if not attack_mask.any():
        attack_mask = y_values == 1

    benign_mask = (y_values == 0) & (predictions == 0)
    if not benign_mask.any():
        benign_mask = y_values == 0

    attack_candidates = np.flatnonzero(attack_mask)
    benign_candidates = np.flatnonzero(benign_mask)

    attack_idx = attack_candidates[np.argmax(attack_proba[attack_candidates])]
    benign_idx = benign_candidates[np.argmin(attack_proba[benign_candidates])]

    return attack_idx, benign_idx, attack_proba, predictions


def save_waterfall_plot(
    shap_attack: np.ndarray,
    base_value: float,
    X_sample: pd.DataFrame,
    row_idx: int,
    output_path: Path,
) -> None:
    explanation = shap.Explanation(
        values=shap_attack[row_idx],
        base_values=base_value,
        data=X_sample.iloc[row_idx].to_numpy(),
        feature_names=list(X_sample.columns),
    )

    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(
        explanation,
        max_display=LOCAL_MAX_DISPLAY,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_dependence_plots(
    shap_attack: np.ndarray,
    X_sample: pd.DataFrame,
    importance_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    selected_features = []

    top_feature = importance_df.iloc[0]["feature"]
    selected_features.append(top_feature)

    if "Destination Port" in X_sample.columns and "Destination Port" not in selected_features:
        selected_features.append("Destination Port")

    for feature in selected_features[:2]:
        safe_name = feature.replace("/", "_").replace(" ", "_")
        output_path = output_dir / f"shap_dependence_{safe_name}.png"

        plt.figure()
        shap.dependence_plot(
            feature,
            shap_attack,
            X_sample,
            interaction_index=None,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


def generate_shap_report(
    model,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_df = build_balanced_sample(test_df)

    X_sample = sample_df.drop(columns=["Label"])
    y_sample = sample_df["Label"].astype(int)

    print("Sample class distribution:")
    print(y_sample.value_counts())

    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap_attack = get_attack_shap_values(shap_values)
    base_value = get_attack_expected_value(explainer.expected_value)

    importance_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "mean_abs_shap": abs(shap_attack).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    importance_path = output_dir / "shap_mean_abs_importance.csv"
    summary_path = output_dir / "shap_summary_attack.png"
    bar_path = output_dir / "shap_bar_attack.png"
    attack_waterfall_path = output_dir / "shap_waterfall_attack_high_confidence.png"
    benign_waterfall_path = output_dir / "shap_waterfall_benign_high_confidence.png"
    examples_path = output_dir / "shap_local_examples.csv"

    importance_df.to_csv(importance_path, index=False)

    plt.figure()
    shap.summary_plot(
        shap_attack,
        X_sample,
        show=False,
        max_display=MAX_DISPLAY,
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_attack,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=MAX_DISPLAY,
    )
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    attack_idx, benign_idx, attack_proba, predictions = choose_local_examples(
        model,
        X_sample,
        y_sample,
    )

    save_waterfall_plot(
        shap_attack,
        base_value,
        X_sample,
        attack_idx,
        attack_waterfall_path,
    )
    save_waterfall_plot(
        shap_attack,
        base_value,
        X_sample,
        benign_idx,
        benign_waterfall_path,
    )
    save_dependence_plots(shap_attack, X_sample, importance_df, output_dir)

    examples_df = pd.DataFrame(
        [
            {
                "example": "attack_high_confidence",
                "sample_row": int(attack_idx),
                "true_label": int(y_sample.iloc[attack_idx]),
                "predicted_label": int(predictions[attack_idx]),
                "attack_probability": float(attack_proba[attack_idx]),
            },
            {
                "example": "benign_high_confidence",
                "sample_row": int(benign_idx),
                "true_label": int(y_sample.iloc[benign_idx]),
                "predicted_label": int(predictions[benign_idx]),
                "attack_probability": float(attack_proba[benign_idx]),
            },
        ]
    )
    examples_df.to_csv(examples_path, index=False)

    print(f"Saved SHAP outputs to: {output_dir}")
    print(importance_df.head(MAX_DISPLAY).to_string(index=False))
    print("\nLocal examples:")
    print(examples_df.to_string(index=False))

    return importance_df, examples_df


def main() -> None:
    args = parse_args()
    paths = get_variant_paths(args.variant)
    output_dir = paths.processed_dir / "shap_random_forest"

    print("Loading model...")
    model = joblib.load(paths.random_forest_model_file)

    print("Loading test data...")
    test_df = pd.read_csv(paths.test_file)
    generate_shap_report(model, test_df, output_dir)


if __name__ == "__main__":
    main()

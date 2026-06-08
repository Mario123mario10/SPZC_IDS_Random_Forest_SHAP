from __future__ import annotations

import argparse

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from variant_paths import add_variant_argument, get_variant_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_variant_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_variant_paths(args.variant)

    model_path = paths.model_dir / "random_forest_without_all_packet.joblib"
    csv_output_path = (
        paths.processed_dir / "random_forest_without_all_packet_feature_importance.csv"
    )
    plot_output_path = (
        paths.processed_dir / "random_forest_without_all_packet_feature_importance_top30.png"
    )

    model = joblib.load(model_path)

    if not hasattr(model, "feature_names_in_"):
        raise AttributeError(
            "Model does not expose feature_names_in_. "
            "It was probably trained on a NumPy array instead of a pandas DataFrame."
        )

    feature_names = model.feature_names_in_
    importances = model.feature_importances_

    print("Number of feature names:", len(feature_names))
    print("Number of importances:", len(importances))

    if len(feature_names) != len(importances):
        raise ValueError(
            f"Mismatch: {len(feature_names)} feature names, but {len(importances)} importances."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(csv_output_path, index=False)

    top_n = 30
    top_df = importance_df.head(top_n).sort_values("importance")

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Random Forest feature importances")
    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=300)

    print(f"Saved CSV to: {csv_output_path}")
    print(f"Saved plot to: {plot_output_path}")

    print("\nTop 30 features:")
    print(importance_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()

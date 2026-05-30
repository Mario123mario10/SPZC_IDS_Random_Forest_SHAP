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

    model_path = paths.model_dir / "random_forest_port_80_only.joblib"
    csv_output_path = paths.processed_dir / "random_forest_port_80_only_feature_importance.csv"
    plot_output_path = (
        paths.processed_dir / "random_forest_port_80_only_feature_importance_top30.png"
    )

    model = joblib.load(model_path)

    feature_names = model.feature_names_in_
    importances = model.feature_importances_

    assert "Destination Port" not in feature_names, (
        "Destination Port is still present in the model features. "
        "It should be removed after filtering the port_80_only subset."
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(csv_output_path, index=False)

    top_df = importance_df.head(30).sort_values("importance")

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Top 30 feature importances - port 80 only")
    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=300)
    plt.close()

    print("\nTop 30 features:")
    print(importance_df.head(30).to_string(index=False))

    print(f"\nSaved CSV to: {csv_output_path}")
    print(f"Saved plot to: {plot_output_path}")


if __name__ == "__main__":
    main()

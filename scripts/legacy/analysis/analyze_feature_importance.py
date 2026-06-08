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

    model_path = paths.random_forest_model_file
    train_path = paths.train_file
    csv_output_path = paths.processed_dir / "random_forest_feature_importance.csv"
    plot_output_path = paths.processed_dir / "random_forest_feature_importance_top30.png"

    model = joblib.load(model_path)

    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["Label"])

    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
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

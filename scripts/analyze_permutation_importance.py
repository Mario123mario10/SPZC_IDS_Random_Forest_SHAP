from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from variant_paths import add_variant_argument, get_variant_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_variant_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_variant_paths(args.variant)

    model_path = paths.random_forest_model_file
    test_path = paths.test_file
    output_path = paths.processed_dir / "random_forest_permutation_importance.csv"

    print("Loading model...")
    model = joblib.load(model_path)

    print("Loading test data...")
    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"].astype(int)

    # Use a sample for speed. The full test.csv also works but may take longer.
    sample_size = min(20000, len(X_test))

    X_sample = X_test.sample(n=sample_size, random_state=42)
    y_sample = y_test.loc[X_sample.index]

    print(f"Calculating permutation importance on {sample_size} samples...")

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring="f1",
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    print("\nTop 30 permutation importances:")
    print(importance_df.head(30).to_string(index=False))

    importance_df.to_csv(output_path, index=False)

    print(f"\nSaved permutation importances to: {output_path}")


if __name__ == "__main__":
    main()

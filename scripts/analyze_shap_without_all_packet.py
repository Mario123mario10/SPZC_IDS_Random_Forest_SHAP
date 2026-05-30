from __future__ import annotations

import argparse

import joblib
import pandas as pd

from analyze_shap_baseline import generate_shap_report
from variant_paths import add_variant_argument, get_variant_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_variant_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_variant_paths(args.variant)
    model_path = paths.model_dir / "random_forest_without_all_packet.joblib"
    output_dir = paths.processed_dir / "shap_without_all_packet"

    print("Loading model...")
    model = joblib.load(model_path)

    if not hasattr(model, "feature_names_in_"):
        raise AttributeError(
            "Model does not expose feature_names_in_. "
            "Retrain it with a pandas DataFrame or pass feature names explicitly."
        )

    feature_names = list(model.feature_names_in_)

    print("Loading test data...")
    test_df = pd.read_csv(paths.test_file)

    missing_features = sorted(set(feature_names) - set(test_df.columns))
    if missing_features:
        raise ValueError(f"Missing features in test.csv: {missing_features}")

    shap_df = test_df.loc[:, feature_names].copy()
    shap_df["Label"] = test_df["Label"].astype(int).values

    print(f"Number of model features: {len(feature_names)}")
    generate_shap_report(model, shap_df, output_dir)


if __name__ == "__main__":
    main()

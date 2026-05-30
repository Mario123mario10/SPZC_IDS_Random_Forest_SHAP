import argparse

import joblib
import pandas as pd

from variant_paths import add_variant_argument, get_variant_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_variant_argument(parser)
    return parser.parse_args()


args = parse_args()
paths = get_variant_paths(args.variant)

scaler = joblib.load(paths.scaler_file)

if hasattr(scaler, "feature_names_in_"):
    feature_names = scaler.feature_names_in_
else:
    feature_names = [f"feature_{i}" for i in range(scaler.n_features_in_)]

scaler_df = pd.DataFrame(
    {
        "feature": feature_names,
        "mean": scaler.mean_,
        "std": scaler.scale_,
        "variance": scaler.var_,
    }
)

print(scaler_df.head(20))

output_path = paths.processed_dir / "standard_scaler_parameters.csv"
scaler_df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")

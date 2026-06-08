import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

from variant_paths import add_variant_argument, get_variant_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_variant_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_variant_paths(args.variant)

    df = pd.read_csv(paths.train_file)

    X = df.drop(columns=["Label"])
    y = df["Label"].astype(int)

    rows = []

    for column in X.columns:
        values = X[column]

        try:
            auc = roc_auc_score(y, values)
            auc = max(auc, 1 - auc)
        except ValueError:
            auc = None

        rows.append(
            {
                "feature": column,
                "single_feature_auc": auc,
            }
        )

    auc_df = pd.DataFrame(rows).sort_values("single_feature_auc", ascending=False)

    print(auc_df.head(30).to_string(index=False))

    output_path = paths.processed_dir / "single_feature_auc.csv"
    auc_df.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

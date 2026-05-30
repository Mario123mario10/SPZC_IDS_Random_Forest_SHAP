import argparse

import pandas as pd

from variant_paths import add_variant_argument, get_variant_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_variant_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_variant_paths(args.variant)

    train_df = pd.read_csv(paths.train_file)
    test_df = pd.read_csv(paths.test_file)

    train_hashes = pd.util.hash_pandas_object(train_df, index=False)
    test_hashes = pd.util.hash_pandas_object(test_df, index=False)

    overlap = set(train_hashes).intersection(set(test_hashes))

    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Exact duplicate rows between train and test:", len(overlap))

    train_feature_hashes = pd.util.hash_pandas_object(
        train_df.drop(columns=["Label"]),
        index=False,
    )
    test_feature_hashes = pd.util.hash_pandas_object(
        test_df.drop(columns=["Label"]),
        index=False,
    )

    feature_overlap = set(train_feature_hashes).intersection(set(test_feature_hashes))

    print("Exact duplicate feature rows between train and test:", len(feature_overlap))


if __name__ == "__main__":
    main()

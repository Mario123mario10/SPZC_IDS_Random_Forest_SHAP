from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data_processed")

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

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
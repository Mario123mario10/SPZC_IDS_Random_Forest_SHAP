from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

PROCESSED_DIR = Path("data_processed")

TRAIN_PATH = PROCESSED_DIR / "train.csv"
OUTPUT_PATH = PROCESSED_DIR / "single_feature_auc.csv"


def main() -> None:
    df = pd.read_csv(TRAIN_PATH)

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

    auc_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_DIR = Path("data_processed")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "random_forest_baseline.joblib"
TRAIN_PATH = PROCESSED_DIR / "train.csv"

CSV_OUTPUT_PATH = PROCESSED_DIR / "random_forest_feature_importance.csv"
PLOT_OUTPUT_PATH = PROCESSED_DIR / "random_forest_feature_importance_top30.png"


def main() -> None:
    model = joblib.load(MODEL_PATH)

    train_df = pd.read_csv(TRAIN_PATH)
    X_train = train_df.drop(columns=["Label"])

    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(CSV_OUTPUT_PATH, index=False)

    top_n = 30
    top_df = importance_df.head(top_n).sort_values("importance")

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Random Forest feature importances")
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)

    print(f"Saved CSV to: {CSV_OUTPUT_PATH}")
    print(f"Saved plot to: {PLOT_OUTPUT_PATH}")

    print("\nTop 30 features:")
    print(importance_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
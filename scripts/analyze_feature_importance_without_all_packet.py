from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_DIR = Path("data_processed")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "random_forest_without_all_packet.joblib"
TRAIN_PATH = PROCESSED_DIR / "train.csv"

CSV_OUTPUT_PATH = PROCESSED_DIR / "random_forest_without_all_packet_feature_importance.csv"
PLOT_OUTPUT_PATH = PROCESSED_DIR / "random_forest_without_all_packet_feature_importance_top30.png"


def main() -> None:
    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "feature_names_in_"):
        raise AttributeError(
            "Model nie ma atrybutu feature_names_in_. "
            "Prawdopodobnie był trenowany na numpy array zamiast pandas DataFrame."
        )

    feature_names = model.feature_names_in_
    importances = model.feature_importances_

    print("Number of feature names:", len(feature_names))
    print("Number of importances:", len(importances))

    if len(feature_names) != len(importances):
        raise ValueError(
            f"Mismatch: {len(feature_names)} feature names, "
            f"but {len(importances)} importances."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
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
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

PROCESSED_DIR = Path("data_processed")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "random_forest_baseline.joblib"
TEST_PATH = PROCESSED_DIR / "test.csv"

OUTPUT_PATH = PROCESSED_DIR / "random_forest_permutation_importance.csv"


def main() -> None:
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"].astype(int)

    # Dla szybkości można wziąć próbkę.
    # Przy pełnym test.csv też zadziała, ale może trwać dłużej.
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

    importance_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved permutation importances to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
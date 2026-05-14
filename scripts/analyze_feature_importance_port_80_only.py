from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_DIR = Path("data_processed")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "random_forest_port_80_only.joblib"

CSV_OUTPUT_PATH = PROCESSED_DIR / "random_forest_port_80_only_feature_importance.csv"
PLOT_OUTPUT_PATH = PROCESSED_DIR / "random_forest_port_80_only_feature_importance_top30.png"


def main() -> None:
    model = joblib.load(MODEL_PATH)

    feature_names = model.feature_names_in_
    importances = model.feature_importances_

    assert "Destination Port" not in feature_names, (
        "Destination Port nadal jest w cechach modelu. "
        "W teście port_80_only powinien być usunięty po filtracji."
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(CSV_OUTPUT_PATH, index=False)

    top_df = importance_df.head(30).sort_values("importance")

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Top 30 feature importances — port 80 only")
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
    plt.close()

    print("\nTop 30 features:")
    print(importance_df.head(30).to_string(index=False))

    print(f"\nSaved CSV to: {CSV_OUTPUT_PATH}")
    print(f"Saved plot to: {PLOT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
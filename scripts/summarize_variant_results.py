from __future__ import annotations

from pathlib import Path

import pandas as pd

REPORTS_TABLE_DIR = Path("reports") / "tables"
BASE_PROCESSED_DIR = Path("data_processed")


def load_metrics(variant_name: str, file_path: Path) -> pd.DataFrame | None:
    if not file_path.exists():
        print(f"Warning: Metrics file not found for '{variant_name}': {file_path}")
        return None

    df = pd.read_csv(file_path)
    df["variant"] = variant_name
    return df


def main() -> None:
    variants_to_summarize = {
        "paper_baseline": BASE_PROCESSED_DIR / "paper_baseline" / "random_forest_metrics.csv",
        "controlled": BASE_PROCESSED_DIR / "controlled" / "random_forest_metrics.csv",
        "main_internal_2017": REPORTS_TABLE_DIR / "internal_2017_metrics.csv",
        "main_external_2018": REPORTS_TABLE_DIR / "external_2018_metrics.csv",
        "generalization_internal_2017": BASE_PROCESSED_DIR
        / "generalization_test"
        / "metrics_internal_2017.csv",
        "generalization_external_2018": BASE_PROCESSED_DIR
        / "generalization_test"
        / "metrics_external_2018.csv",
    }

    all_metrics = []
    for name, path in variants_to_summarize.items():
        metrics_df = load_metrics(name, path)
        if metrics_df is not None:
            all_metrics.append(metrics_df)

    if not all_metrics:
        print("No metric files found. Run experiments first ('just main', 'just paper', etc.).")
        return

    summary_df = pd.concat(all_metrics, ignore_index=True)

    cols = ["variant", "accuracy", "precision", "recall", "f1_score", "samples"]
    existing_cols = [c for c in cols if c in summary_df.columns]
    summary_df = summary_df[existing_cols]

    output_csv_path = REPORTS_TABLE_DIR / "variants_summary.csv"
    REPORTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv_path, index=False, float_format="%.4f")
    print(f"\nSummary of results saved to: {output_csv_path}")

    print("\n--- Results Summary (Markdown) ---")
    print(summary_df.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()

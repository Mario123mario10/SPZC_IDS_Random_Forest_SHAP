from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data_raw")
PROCESSED_DIR = Path("data_processed")

def count_infinite_values(df: pd.DataFrame) -> pd.Series:
    """Zwraca liczbę wartości inf/-inf w kolumnach numerycznych."""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return pd.Series(dtype="int64")

    return np.isinf(numeric_df).sum()


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Czyści DataFrame i zwraca oczyszczone dane oraz statystyki."""
    rows_before = len(df)

    # CICIDS2017 często ma spacje na początku nazw kolumn, np. " Flow Packets/s".
    df = df.copy()
    df.columns = df.columns.str.strip()

    missing_before = int(df.isna().sum().sum())
    infinite_before = int(count_infinite_values(df).sum())

    # Zamiana wartości nieskończonych na brakujące.
    df = df.replace([np.inf], np.nan)

    # Usunięcie wszystkich rekordów z brakami.
    df_clean = df.dropna().reset_index(drop=True)

    rows_after = len(df_clean)
    missing_after = int(df_clean.isna().sum().sum())
    infinite_after = int(count_infinite_values(df_clean).sum())

    stats = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "removed_rows": rows_before - rows_after,
        "missing_before": missing_before,
        "infinite_before": infinite_before,
        "missing_after": missing_after,
        "infinite_after": infinite_after,
    }

    return df_clean, stats


def preprocess_file(csv_path: Path) -> dict[str, object]:
    """Czyści pojedynczy plik CSV i zapisuje wynik w data_processed/."""
    print(f"\nProcessing: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df_clean, stats = clean_dataframe(df)

    output_path = PROCESSED_DIR / f"{csv_path.stem}_clean.csv"
    df_clean.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows before: {stats['rows_before']}")
    print(f"Rows after:  {stats['rows_after']}")
    print(f"Removed:     {stats['removed_rows']}")
    print(f"Missing after cleaning:  {stats['missing_after']}")
    print(f"Infinite after cleaning: {stats['infinite_after']}")

    return {
        "file": csv_path.name,
        "output_file": output_path.name,
        **stats,
    }


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Brakuje katalogu {RAW_DIR}. Utwórz go i wrzuć tam surowe pliki CSV."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Nie znaleziono plików CSV w katalogu {RAW_DIR}.")

    reports = []

    for csv_path in csv_files:
        reports.append(preprocess_file(csv_path))

    report_df = pd.DataFrame(reports)
    report_path = PROCESSED_DIR / "cleaning_report.csv"
    report_df.to_csv(report_path, index=False)

    print("\nPreprocessing finished.")
    print(f"Cleaning report saved to: {report_path}")

    # Dodatkowe zabezpieczenie: jeśli po czyszczeniu zostały braki/inf, zatrzymaj skrypt.
    total_missing_after = int(report_df["missing_after"].sum())
    total_infinite_after = int(report_df["infinite_after"].sum())

    assert total_missing_after == 0, "Po czyszczeniu nadal istnieją brakujące wartości."
    assert total_infinite_after == 0, "Po czyszczeniu nadal istnieją wartości nieskończone."


if __name__ == "__main__":
    main()

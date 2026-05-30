from __future__ import annotations


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import joblib
from sklearn.preprocessing import StandardScaler

from variant_paths import BASE_PROCESSED_DIR, get_variant_paths


RAW_DIR = Path("data_raw")
VARIANT = "paper_baseline"
PATHS = get_variant_paths(VARIANT)
CLEANED_DIR = BASE_PROCESSED_DIR
PROCESSED_DIR = PATHS.processed_dir

MONDAY_CLEAN_FILE = CLEANED_DIR / "Monday-WorkingHours.pcap_ISCX_clean.csv"
FRIDAY_DDOS_CLEAN_FILE = CLEANED_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_clean.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def count_infinite_values(df: pd.DataFrame) -> pd.Series:
    """Return the number of inf/-inf values in numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return pd.Series(dtype="int64")

    return np.isinf(numeric_df).sum()


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean a DataFrame and return cleaned data with cleaning statistics."""
    rows_before = len(df)

    # CICIDS2017 often has leading spaces in column names, e.g. " Flow Packets/s".
    df = df.copy()
    df.columns = df.columns.str.strip()

    missing_before = int(df.isna().sum().sum())
    infinite_before = int(count_infinite_values(df).sum())

    # Convert infinite values to missing values.
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove all rows with missing values.
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
    """Clean one CSV file and write the result to data_processed/."""
    print(f"\nProcessing: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df_clean, stats = clean_dataframe(df)

    output_path = CLEANED_DIR / f"{csv_path.stem}_clean.csv"
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


def load_baseline_dataset() -> pd.DataFrame:
    """Create the paper-like baseline dataset: Monday benign + Friday DDoS."""
    if not MONDAY_CLEAN_FILE.exists():
        raise FileNotFoundError(f"File not found: {MONDAY_CLEAN_FILE}")

    if not FRIDAY_DDOS_CLEAN_FILE.exists():
        raise FileNotFoundError(f"File not found: {FRIDAY_DDOS_CLEAN_FILE}")

    monday_df = pd.read_csv(MONDAY_CLEAN_FILE, low_memory=False)
    friday_df = pd.read_csv(FRIDAY_DDOS_CLEAN_FILE, low_memory=False)

    monday_df.columns = monday_df.columns.str.strip()
    friday_df.columns = friday_df.columns.str.strip()

    if "Label" not in monday_df.columns:
        raise KeyError(f"Missing Label column in file {MONDAY_CLEAN_FILE}")

    if "Label" not in friday_df.columns:
        raise KeyError(f"Missing Label column in file {FRIDAY_DDOS_CLEAN_FILE}")

    monday_df["Label"] = monday_df["Label"].astype(str).str.strip()
    friday_df["Label"] = friday_df["Label"].astype(str).str.strip()

    # Monday contains only benign traffic for the paper-like baseline.
    monday_benign_df = monday_df[monday_df["Label"].str.lower() == "benign"].copy()

    # Friday DDoS file contributes only DDoS records in the paper-like baseline.
    friday_ddos_df = friday_df[friday_df["Label"].str.lower() == "ddos"].copy()

    if monday_benign_df.empty:
        raise ValueError("No Benign records found after filtering Monday data.")

    if friday_ddos_df.empty:
        raise ValueError("No DDoS records found after filtering Friday data.")

    # Normalize label names.
    monday_benign_df["Label"] = "Benign"
    friday_ddos_df["Label"] = "DDoS"

    combined_df = pd.concat([monday_benign_df, friday_ddos_df], ignore_index=True)

    print("\nClass counts in the baseline dataset:")
    print(combined_df["Label"].value_counts())

    combined_df.to_csv(PROCESSED_DIR / "baseline_ddos_dataset.csv", index=False)
    combined_df["Label"].value_counts().to_csv(PROCESSED_DIR / "baseline_class_distribution.csv")

    return combined_df


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Missing {RAW_DIR} directory. Create it and place the raw CSV files there."
        )

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}.")

    reports = []

    for csv_path in csv_files:
        result = preprocess_file(csv_path)
        reports.append(result)

    report_df = pd.DataFrame(reports)
    report_path = CLEANED_DIR / "cleaning_report.csv"
    report_df.to_csv(report_path, index=False)

    # Safety check: stop if missing or infinite values remain after cleaning.
    total_missing_after = int(report_df["missing_after"].sum())
    total_infinite_after = int(report_df["infinite_after"].sum())

    assert total_missing_after == 0, "Missing values remain after cleaning."
    assert total_infinite_after == 0, "Infinite values remain after cleaning."

    print(f"Cleaning report saved to: {report_path}")

    print("\nCreating baseline dataset: Monday Benign + Friday DDoS")
    combined_df = load_baseline_dataset()
    print(f"Baseline dataset shape: {combined_df.shape}")

    print("Encoding labels: Benign -> 0, Attack/DDoS -> 1")

    combined_df["Label"] = combined_df["Label"].astype(str).str.strip()

    y = combined_df["Label"].apply(lambda label: 0 if label.lower() == "benign" else 1)

    X = combined_df.drop(columns=["Label"])

    # Safety check: Random Forest and StandardScaler expect numeric features.
    non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            "The feature set contains non-numeric columns: "
            f"{non_numeric_columns}. Remove or encode them before scaling."
        )

    print("Splitting into train and test sets")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Scaling features with StandardScaler")

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    joblib.dump(scaler, PROCESSED_DIR / "standard_scaler.joblib")

    train_df = X_train_scaled.copy()
    train_df["Label"] = y_train.values
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)

    test_df = X_test_scaled.copy()
    test_df["Label"] = y_test.values
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"Saved train.csv and test.csv in {PROCESSED_DIR}")
    print(f"{PROCESSED_DIR / 'train.csv'}: {train_df.shape}")
    print(f"{PROCESSED_DIR / 'test.csv'}: {test_df.shape}")

    print("\nClass distribution in train:")
    print(train_df["Label"].value_counts())

    print("\nClass distribution in test:")
    print(test_df["Label"].value_counts())

    print("\nPreprocessing finished.")


if __name__ == "__main__":
    main()

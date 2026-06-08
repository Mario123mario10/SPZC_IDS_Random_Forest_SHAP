from __future__ import annotations


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import joblib
from sklearn.preprocessing import StandardScaler

from variant_paths import get_variant_paths


RAW_DIR = Path("data_raw")
VARIANT = "paper_baseline"
PATHS = get_variant_paths(VARIANT)
PROCESSED_DIR = PATHS.processed_dir

TEST_SIZE = 0.2
RANDOM_STATE = 42


def count_infinite_values(df: pd.DataFrame) -> pd.Series:
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return pd.Series(dtype="int64")

    return np.isinf(numeric_df).sum()


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows_before = len(df)

    df = df.copy()
    df.columns = df.columns.str.strip()

    missing_before = int(df.isna().sum().sum())
    infinite_before = int(count_infinite_values(df).sum())

    df = df.replace([np.inf, -np.inf], np.nan)

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


def load_baseline_dataset(monday_df: pd.DataFrame, friday_df: pd.DataFrame) -> pd.DataFrame:
    monday_df.columns = monday_df.columns.str.strip()
    friday_df.columns = friday_df.columns.str.strip()

    if "Label" not in monday_df.columns:
        raise KeyError("Missing Label column in Monday data.")

    if "Label" not in friday_df.columns:
        raise KeyError("Missing Label column in Friday DDoS data.")

    monday_df["Label"] = monday_df["Label"].astype(str).str.strip()
    friday_df["Label"] = friday_df["Label"].astype(str).str.strip()

    monday_benign_df = monday_df[monday_df["Label"].str.lower() == "benign"].copy()

    friday_ddos_df = friday_df[friday_df["Label"].str.lower() == "ddos"].copy()

    if monday_benign_df.empty:
        raise ValueError("No Benign records found after filtering Monday data.")

    if friday_ddos_df.empty:
        raise ValueError("No DDoS records found after filtering Friday data.")

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

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cic_dir = RAW_DIR / "CIC_IDS2017"
    if not cic_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {cic_dir}. "
            "Ensure CIC-IDS-2017 files are in the correct subdirectory."
        )

    all_found_files = list(cic_dir.glob("*.csv"))

    monday_file = next(
        (p for p in all_found_files if "monday-workinghours" in p.name.lower()), None
    )
    friday_file = next(
        (p for p in all_found_files if "friday-workinghours-afternoon-ddos" in p.name.lower()), None
    )

    if not monday_file or not friday_file:
        missing = []
        if not monday_file:
            missing.append("Monday-WorkingHours*.csv")
        if not friday_file:
            missing.append("Friday-WorkingHours-Afternoon-DDos*.csv")
        raise FileNotFoundError(
            f"Could not find all required files in {cic_dir}. Missing: {missing}"
        )

    csv_files_to_process = [monday_file, friday_file]
    print(f"Processing required files for paper baseline: {[p.name for p in csv_files_to_process]}")

    print(f"\nProcessing: {monday_file.name}")
    monday_raw_df = pd.read_csv(monday_file, low_memory=False, encoding="latin1")
    monday_df_clean, monday_stats = clean_dataframe(monday_raw_df)
    print(f"Rows before: {monday_stats['rows_before']}, Rows after: {monday_stats['rows_after']}")

    print(f"\nProcessing: {friday_file.name}")
    friday_raw_df = pd.read_csv(friday_file, low_memory=False, encoding="latin1")
    friday_df_clean, friday_stats = clean_dataframe(friday_raw_df)
    print(f"Rows before: {friday_stats['rows_before']}, Rows after: {friday_stats['rows_after']}")

    reports = []
    reports.append({"file": monday_file.name, **monday_stats})
    reports.append({"file": friday_file.name, **friday_stats})

    report_df = pd.DataFrame(reports)
    report_path = PROCESSED_DIR / "cleaning_report.csv"
    report_df.to_csv(report_path, index=False)

    print(f"\nCleaning report saved to: {report_path}")

    print("\nCreating baseline dataset: Monday Benign + Friday DDoS")
    combined_df = load_baseline_dataset(monday_df_clean, friday_df_clean)
    print(f"Baseline dataset shape: {combined_df.shape}")

    print("Encoding labels: Benign -> 0, Attack/DDoS -> 1")

    combined_df["Label"] = combined_df["Label"].astype(str).str.strip()

    y = combined_df["Label"].apply(lambda label: 0 if label.lower() == "benign" else 1)

    X = combined_df.drop(columns=["Label"])

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

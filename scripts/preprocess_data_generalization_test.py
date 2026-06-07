from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

from variant_paths import get_variant_paths

RAW_DIR = Path("data_raw")
VARIANT = "generalization_test"
PATHS = get_variant_paths(VARIANT)
PROCESSED_DIR = PATHS.processed_dir
CLEANED_DATA_DIR = PATHS.cleaned_data_dir

TEST_SIZE = 0.2
RANDOM_STATE = 42

def count_infinite_values(df: pd.DataFrame) -> pd.Series:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.Series(dtype="int64")
    return np.isinf(numeric_df).sum()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean


def load_and_clean_files(file_list: list[str]) -> pd.DataFrame:
    all_df = []
    print(f"Processing {len(file_list)} files...")
    for filename in tqdm(file_list, desc="Loading and cleaning files"):
        path = RAW_DIR / filename
        if not path.exists():
            print(f"Warning: File not found, skipping: {path}")
            continue
        try:
            df = pd.read_csv(path, low_memory=False, encoding="latin1")
            df_clean = clean_dataframe(df)
            all_df.append(df_clean)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if not all_df:
        raise ValueError("No data could be loaded. Check file paths and contents.")

    return pd.concat(all_df, ignore_index=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for dataset files in {RAW_DIR}...")
    all_raw_files = [p.name for p in RAW_DIR.glob("*.csv")]

    files_2017 = sorted([f for f in all_raw_files if f.endswith(".pcap_ISCX.csv")])
    files_2018 = sorted([f for f in all_raw_files if re.match(r"\d{2}-\d{2}-\d{4}\.csv", f)])

    if not files_2017:
        raise FileNotFoundError(f"No CIC-IDS-2017 files (*.pcap_ISCX.csv) found in '{RAW_DIR}'.")
    if not files_2018:
        raise FileNotFoundError(
            f"No CSE-CIC-IDS-2018 files (DD-MM-YYYY.csv) found in '{RAW_DIR}'. "
            "This is required for the generalization test."
        )

    print(f"Found {len(files_2017)} CIC-IDS-2017 files.")
    print(f"Found {len(files_2018)} CSE-CIC-IDS-2018 files.")

    print("--- Processing CIC-IDS-2017 Dataset ---")
    df_2017 = load_and_clean_files(files_2017)
    df_2017 = df_2017.drop_duplicates(ignore_index=True)

    print("\n--- Processing CSE-CIC-IDS-2018 Dataset ---")
    df_2018 = load_and_clean_files(files_2018)
    df_2018 = df_2018.drop_duplicates(ignore_index=True)

    df_2017.columns = df_2017.columns.str.strip().str.lower().str.replace(" ", "_")
    df_2018.columns = df_2018.columns.str.strip().str.lower().str.replace(" ", "_")

    common_cols = sorted(list(set(df_2017.columns) & set(df_2018.columns)))

    if "label" not in common_cols:
        raise KeyError("The 'Label' column is not common to both datasets after cleaning.")

    df_2017 = df_2017[common_cols]
    df_2018 = df_2018[common_cols]
    print(f"\nFound {len(common_cols)} common features between datasets.")

    y_2017 = df_2017["label"].apply(lambda label: 0 if str(label).strip().upper() == "BENIGN" else 1)
    X_2017 = df_2017.drop(columns=["label"])

    y_2018 = df_2018["label"].apply(lambda label: 0 if str(label).strip().upper() == "BENIGN" else 1)
    X_2018 = df_2018.drop(columns=["label"])

    non_numeric_cols = X_2017.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
        X_2017 = X_2017.drop(columns=non_numeric_cols)
        X_2018 = X_2018.drop(columns=non_numeric_cols)

    print("\nSplitting 2017 data into train and internal test sets...")
    X_train, X_test_internal, y_train, y_test_internal = train_test_split(
        X_2017, y_2017, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_2017
    )

    X_test_external = X_2018
    y_test_external = y_2018

    print("\nFitting scaler on training data and transforming all sets...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_internal_scaled = pd.DataFrame(scaler.transform(X_test_internal), columns=X_test_internal.columns, index=X_test_internal.index)
    X_test_external_scaled = pd.DataFrame(scaler.transform(X_test_external), columns=X_test_external.columns, index=X_test_external.index)

    joblib.dump(scaler, PROCESSED_DIR / "standard_scaler.joblib")

    train_df = X_train_scaled.copy()
    train_df["Label"] = y_train.values
    train_df.to_csv(PATHS.train_file, index=False)
    print(f"Saved training set: {PATHS.train_file} (Shape: {train_df.shape})")

    test_internal_df = X_test_internal_scaled.copy()
    test_internal_df["Label"] = y_test_internal.values
    test_internal_df.to_csv(PATHS.test_file, index=False)
    print(f"Saved internal test set: {PATHS.test_file} (Shape: {test_internal_df.shape})")

    test_external_df = X_test_external_scaled.copy()
    test_external_df["Label"] = y_test_external.values
    test_external_df.to_csv(PATHS.test_file_external, index=False)
    print(f"Saved external test set: {PATHS.test_file_external} (Shape: {test_external_df.shape})")

    print("\nPreprocessing for generalization test finished.")


if __name__ == "__main__":
    main()

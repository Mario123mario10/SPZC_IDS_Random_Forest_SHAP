from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

from variant_paths import get_variant_paths

RAW_DIR = Path("data_raw")

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


def load_and_label_portscan_data(cleaned_files: list[Path], processed_dir: Path) -> pd.DataFrame:
    all_df = []
    if not cleaned_files:
        raise ValueError("No cleaned files provided to load.")

    for csv_path in cleaned_files:
        if not csv_path.exists():
            raise FileNotFoundError(f"Cleaned file not found: {csv_path}")
        print(f"Loading cleaned file: {csv_path.name}")
        df = pd.read_csv(csv_path, low_memory=False)
        all_df.append(df)

    combined_df = pd.concat(all_df, ignore_index=True)
    combined_df.columns = combined_df.columns.str.strip()

    portscan_labels = ["portscan", "port scan"]

    def map_label(label):
        processed_label = str(label).strip().lower()
        if processed_label in portscan_labels:
            return "PortScan"
        elif processed_label == "benign":
            return "Benign"
        else:
            return "Other"

    combined_df["Label"] = combined_df["Label"].apply(map_label)

    combined_df = combined_df[combined_df["Label"].isin(["Benign", "PortScan"])].copy()

    print("\nClass counts in the PortScan dataset:")
    print(combined_df["Label"].value_counts())

    if combined_df["Label"].nunique() < 2:
        raise ValueError(
            "Only one class found after labeling. For binary classification, "
            "please provide files containing both 'Benign' traffic and 'PortScan' attack traffic."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(processed_dir / "portscan_dataset.csv", index=False)
    combined_df["Label"].value_counts().to_csv(processed_dir / "portscan_class_distribution.csv")

    return combined_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess data for PortScan detection.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "data_raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "data_raw/Monday-WorkingHours.pcap_ISCX.csv",
        ],
        help="Paths to raw CSV files to process. Should include attack and benign traffic.",
    )
    args = parser.parse_args()

    variant = "portscan"
    PATHS = get_variant_paths(variant)
    PROCESSED_DIR = PATHS.processed_dir
    CLEANED_DATA_DIR = PATHS.cleaned_data_dir

    csv_files = [Path(f) for f in args.files]
    for f in csv_files:
        if not f.exists():
            raise FileNotFoundError(
                f"Input file not found: {f}\n"
                f"Please check the path is correct relative to the current directory: {Path.cwd()}"
            )

    processing_source_str = f"specific files: {[str(f) for f in csv_files]}"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not csv_files:
        raise FileNotFoundError("No CSV files found for processing.")

    cleaned_file_paths = []
    reports = []
    print(
        f"Cleaning raw CSV files from {processing_source_str} and saving to {CLEANED_DATA_DIR}..."
    )
    for csv_path in tqdm(csv_files, desc="Cleaning files"):
        output_cleaned_file_path = CLEANED_DATA_DIR / f"{csv_path.stem}_clean.csv"
        df = pd.read_csv(csv_path, low_memory=False)
        df_clean, stats = clean_dataframe(df)
        df_clean.to_csv(output_cleaned_file_path, index=False)
        reports.append(
            {"file": csv_path.name, "output_file": output_cleaned_file_path.name, **stats}
        )
        cleaned_file_paths.append(output_cleaned_file_path)

    report_df = pd.DataFrame(reports)
    report_path = PATHS.cleaning_report_file
    report_df.to_csv(report_path, index=False)
    print(f"Cleaning report saved to: {report_path}")

    print(f"Loading and labeling data for {variant}...")
    combined_df = load_and_label_portscan_data(cleaned_file_paths, PROCESSED_DIR)
    print(f"{variant} dataset shape: {combined_df.shape}")

    print("Encoding labels: Benign -> 0, Attack -> 1")
    y = combined_df["Label"].apply(lambda label: 0 if label.lower() == "benign" else 1)
    X = combined_df.drop(columns=["Label"])

    non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"Non-numeric columns in X: {non_numeric_columns}")

    print("Splitting into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Scaling features with StandardScaler")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
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

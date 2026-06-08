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


def process_and_combine_data(dataframes: list[pd.DataFrame], processed_dir: Path) -> pd.DataFrame:
    if not dataframes:
        raise ValueError("No dataframes provided to process.")

    for df in dataframes:
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if not dataframes:
        raise ValueError("No dataframes were loaded.")

    common_cols = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_cols.intersection_update(df.columns)

    common_cols = sorted(list(common_cols))
    if "label" not in common_cols:
        raise KeyError("The 'label' column is not common to all datasets after cleaning.")

    print(f"\nFound {len(common_cols)} common features across all files.")

    all_df_common = [df[common_cols] for df in dataframes]
    combined_df = pd.concat(all_df_common, ignore_index=True)

    combined_df["original_label"] = combined_df["label"].str.strip()

    bruteforce_labels = [
        "ftp-patator",
        "ssh-patator",
        "ftp-bruteforce",
        "ssh-bruteforce",
    ]

    def map_label(label):
        if str(label).strip().lower() in bruteforce_labels:
            return "BruteForce"
        elif str(label).strip().lower() == "benign":
            return "Benign"
        else:
            return "Other"

    combined_df["label"] = combined_df["label"].apply(map_label)

    combined_df = combined_df[combined_df["label"].isin(["Benign", "BruteForce"])].copy()

    print("\nClass counts in the BruteForce dataset:")
    print(combined_df["label"].value_counts())

    if combined_df["label"].nunique() < 2:
        raise ValueError(
            "Only one class found after labeling. For binary classification, "
            "please provide files containing both 'Benign' traffic and 'BruteForce' attack traffic."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(processed_dir / "bruteforce_dataset.csv", index=False)
    combined_df["label"].value_counts().to_csv(processed_dir / "bruteforce_class_distribution.csv")

    return combined_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess data for BruteForce detection.")

    default_files = [
        "data_raw/Friday-WorkingHours-Afternoon-Patator.pcap_ISCX.csv",
        "data_raw/02-14-2018.csv",
        "data_raw/02-20-2018.csv",
        "data_raw/Monday-WorkingHours.pcap_ISCX.csv",
    ]

    existing_default_files = [f for f in default_files if Path(f).exists()]
    if not existing_default_files:
        print(
            "Warning: No default data files found. Please specify files using the --files argument."
        )

    parser.add_argument(
        "--files",
        nargs="+",
        default=existing_default_files,
        help="Paths to raw CSV files to process. Should include attack and benign traffic.",
    )
    args = parser.parse_args()

    variant = "bruteforce"
    PATHS = get_variant_paths(variant)
    PROCESSED_DIR = PATHS.processed_dir
    csv_files = [Path(f) for f in args.files]

    attack_default_files = {
        "data_raw/Friday-WorkingHours-Afternoon-Patator.pcap_ISCX.csv",
        "data_raw/02-14-2018.csv",
        "data_raw/02-20-2018.csv",
    }
    used_files_str = {str(f) for f in csv_files}
    has_attack_file = any(f in used_files_str for f in attack_default_files)

    if not csv_files or not has_attack_file:
        missing_files = attack_default_files - used_files_str
        raise FileNotFoundError(
            "Could not find the required BruteForce attack files in your 'data_raw/' directory.\n"
            "The script expects at least one of the following attack files to be present:\n"
            + "\n".join(f"- {f}" for f in sorted(attack_default_files))
            + "\n\nBased on your current files, these seem to be missing:\n"
            + "\n".join(f"- {f}" for f in sorted(missing_files))
        )

    processing_source_str = f"specific files: {[str(f) for f in csv_files]}"

    if not csv_files:
        raise FileNotFoundError("No CSV files found for processing.")

    cleaned_dfs = []
    reports = []
    print(f"Loading and cleaning raw CSV files from {processing_source_str}...")
    for csv_path in tqdm(csv_files, desc="Cleaning files"):
        try:
            df = pd.read_csv(csv_path, low_memory=False, encoding="latin1")
            df_clean, stats = clean_dataframe(df)
            reports.append({"file": csv_path.name, **stats})
            cleaned_dfs.append(df_clean)
        except Exception as e:
            print(f"Could not process {csv_path}: {e}")

    report_df = pd.DataFrame(reports)
    report_path = PROCESSED_DIR / "cleaning_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"Cleaning report saved to: {report_path}")

    print(f"Combining and labeling data for {variant}...")
    combined_df = process_and_combine_data(cleaned_dfs, PROCESSED_DIR)
    print(f"{variant} dataset shape: {combined_df.shape}")

    print("Encoding labels: Benign -> 0, Attack -> 1")
    y = combined_df["label"].apply(lambda label: 0 if label.lower() == "benign" else 1)
    original_labels = combined_df["original_label"]
    X = combined_df.drop(columns=["label", "original_label"])

    non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        print(f"Dropping non-numeric identifier columns: {non_numeric_columns}")
        X = X.drop(columns=non_numeric_columns)

    print("Splitting into train and test sets")
    X_train, X_test, y_train, y_test, _, y_test_original = train_test_split(
        X, y, original_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
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
    test_df["Original_Label"] = y_test_original.values
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

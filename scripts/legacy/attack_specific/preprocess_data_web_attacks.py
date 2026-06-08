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

    def map_label(label):
        processed_label = str(label).strip().lower()
        if processed_label == "benign":
            return "Benign"

        is_web_attack_keyword = "web attack" in processed_label
        is_brute_force_keyword = "brute force" in processed_label
        is_xss_keyword = "xss" in processed_label

        is_bruteforce_web_keyword = "bruteforce" in processed_label and "web" in processed_label
        is_bruteforce_xss_keyword = "bruteforce" in processed_label and "xss" in processed_label

        if is_web_attack_keyword and (is_brute_force_keyword or is_xss_keyword):
            return "WebAttack"

        if is_bruteforce_web_keyword or is_bruteforce_xss_keyword:
            return "WebAttack"

        return "Other"

    combined_df["label"] = combined_df["original_label"].apply(map_label)

    combined_df = combined_df[combined_df["label"].isin(["Benign", "WebAttack"])].copy()

    print("\nClass counts in the WebAttack dataset:")
    print(combined_df["label"].value_counts())

    if combined_df["label"].nunique() < 2:
        raise ValueError(
            "Only one class found after labeling. For binary classification, "
            "please provide files containing both 'Benign' traffic and 'WebAttack' attack traffic."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(processed_dir / "web_attacks_dataset.csv", index=False)
    combined_df["label"].value_counts().to_csv(processed_dir / "web_attacks_class_distribution.csv")

    return combined_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess data for Web Attack detection.")

    default_files = [
        "data_raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "data_raw/02-21-2018.csv",
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

    variant = "web_attacks"
    PATHS = get_variant_paths(variant)
    PROCESSED_DIR = PATHS.processed_dir
    csv_files = [Path(f) for f in args.files]
    for f in csv_files:
        if not f.exists():
            raise FileNotFoundError(
                f"Input file not found: {f}\n"
                f"Please check the path is correct relative to the current directory: {Path.cwd()}"
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

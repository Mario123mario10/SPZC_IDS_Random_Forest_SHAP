from __future__ import annotations

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from binary_pipeline_common import (
    CICIDS2017_BINARY_DIR,
    CICIDS2017_FEATURE_NAMES_FILE,
    CICIDS2017_RAW_DIR,
    CICIDS2017_SCALER_FILE,
    CICIDS2017_TEST_FILE,
    CICIDS2017_TEST_METADATA_FILE,
    CICIDS2017_TRAIN_FILE,
    CICIDS2017_TRAIN_METADATA_FILE,
    MAX_SAMPLES_PER_SOURCE_LABEL,
    RANDOM_STATE,
    TEST_SIZE,
    add_row_ids,
    ensure_dirs,
    load_prepared_directory,
    sample_by_source_label,
    save_distribution_tables,
    save_feature_names,
    split_features_metadata,
)


def choose_stratification(y: pd.Series, metadata: pd.DataFrame) -> pd.Series:
    family_counts = metadata["attack_family"].value_counts()
    if (family_counts >= 2).all():
        return metadata["attack_family"]
    return y


def main() -> None:
    ensure_dirs(CICIDS2017_BINARY_DIR)

    print(f"Loading CICIDS2017 files from {CICIDS2017_RAW_DIR}...")
    combined_df, cleaning_report = load_prepared_directory(CICIDS2017_RAW_DIR, "CICIDS2017")
    cleaning_report.to_csv(CICIDS2017_BINARY_DIR / "cleaning_report.csv", index=False)

    print(f"Rows after cleaning and label filtering: {len(combined_df)}")
    save_distribution_tables(combined_df, CICIDS2017_BINARY_DIR, "before_sampling")

    print(
        "Sampling each source/original-label group to a maximum of "
        f"{MAX_SAMPLES_PER_SOURCE_LABEL} rows..."
    )
    sampled_df = sample_by_source_label(
        combined_df,
        max_samples=MAX_SAMPLES_PER_SOURCE_LABEL,
        random_state=RANDOM_STATE,
    )
    save_distribution_tables(sampled_df, CICIDS2017_BINARY_DIR, "after_sampling")

    X, y, metadata = split_features_metadata(sampled_df)
    feature_names = list(X.columns)
    save_feature_names(feature_names, CICIDS2017_FEATURE_NAMES_FILE)

    print(f"Using {len(feature_names)} numeric features.")
    print("Splitting CICIDS2017 into train/internal-test sets...")
    stratify = choose_stratification(y, metadata)
    X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
        X,
        y,
        metadata,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    print("Fitting StandardScaler on CICIDS2017 train only...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )
    joblib.dump(scaler, CICIDS2017_SCALER_FILE)

    train_df = X_train_scaled.reset_index(drop=True)
    train_df["Label"] = y_train.reset_index(drop=True)
    train_df.to_csv(CICIDS2017_TRAIN_FILE, index=False)

    test_df = X_test_scaled.reset_index(drop=True)
    test_df["Label"] = y_test.reset_index(drop=True)
    test_df.to_csv(CICIDS2017_TEST_FILE, index=False)

    add_row_ids(metadata_train).to_csv(CICIDS2017_TRAIN_METADATA_FILE, index=False)
    add_row_ids(metadata_test).to_csv(CICIDS2017_TEST_METADATA_FILE, index=False)

    print(f"Saved train set: {CICIDS2017_TRAIN_FILE} ({train_df.shape})")
    print(f"Saved internal test set: {CICIDS2017_TEST_FILE} ({test_df.shape})")
    print(f"Saved scaler: {CICIDS2017_SCALER_FILE}")
    print(f"Saved feature names: {CICIDS2017_FEATURE_NAMES_FILE}")
    print("CICIDS2017 binary preprocessing finished.")


if __name__ == "__main__":
    main()

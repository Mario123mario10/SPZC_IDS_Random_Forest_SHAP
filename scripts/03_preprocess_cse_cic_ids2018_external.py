from __future__ import annotations

from collections import Counter

import joblib
import numpy as np
import pandas as pd

from binary_pipeline_common import (
    CICIDS2017_SCALER_FILE,
    CSE_CIC_IDS2018_EXTERNAL_DIR,
    CSE_CIC_IDS2018_RAW_DIR,
    EXTERNAL_METADATA_FILE,
    EXTERNAL_TEST_FILE,
    INCLUDED_ATTACK_FAMILIES,
    binary_label_from_family,
    canonicalize_columns,
    ensure_dirs,
    label_to_attack_family,
    load_feature_names,
)

CHUNK_SIZE = 200_000


def append_csv(df: pd.DataFrame, path, *, header: bool) -> None:
    df.to_csv(path, mode="w" if header else "a", index=False, header=header)


def write_count_table(counter: Counter, index_name: str, output_path) -> None:
    (
        pd.DataFrame(counter.items(), columns=[index_name, "count"])
        .sort_values("count", ascending=False)
        .to_csv(output_path, index=False)
    )


def main() -> None:
    ensure_dirs(CSE_CIC_IDS2018_EXTERNAL_DIR)

    if not CICIDS2017_SCALER_FILE.exists():
        raise FileNotFoundError(
            f"Scaler not found: {CICIDS2017_SCALER_FILE}. "
            "Run scripts/01_preprocess_cicids2017_binary.py first."
        )

    feature_names = load_feature_names()
    scaler = joblib.load(CICIDS2017_SCALER_FILE)

    csv_files = sorted(CSE_CIC_IDS2018_RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {CSE_CIC_IDS2018_RAW_DIR}")

    for output_file in [EXTERNAL_TEST_FILE, EXTERNAL_METADATA_FILE]:
        if output_file.exists():
            output_file.unlink()

    cleaning_reports: list[dict[str, object]] = []
    class_counts: Counter[int] = Counter()
    family_counts: Counter[str] = Counter()
    original_label_counts: Counter[tuple[str, str]] = Counter()

    row_id_start = 0
    write_data_header = True
    write_metadata_header = True

    print(f"Loading CSE-CIC-IDS2018 files from {CSE_CIC_IDS2018_RAW_DIR}...")
    for csv_path in csv_files:
        print(f"Processing {csv_path.name}...")
        rows_before = 0
        rows_after_label_filter = 0
        rows_after_cleaning = 0
        duplicate_columns: set[str] = set()
        raw_row_offset = 0

        for chunk in pd.read_csv(
            csv_path,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding="latin1",
        ):
            chunk_rows = len(chunk)
            rows_before += chunk_rows
            df, duplicates = canonicalize_columns(chunk)
            duplicate_columns.update(duplicates)

            if "label" not in df.columns:
                raise KeyError(f"Missing label column in {csv_path}")

            missing_features = [feature for feature in feature_names if feature not in df.columns]
            if missing_features:
                raise ValueError(
                    f"{csv_path.name} is missing {len(missing_features)} model features: "
                    f"{missing_features}"
                )

            df["source_row"] = np.arange(raw_row_offset, raw_row_offset + len(df))
            raw_row_offset += chunk_rows
            df["original_label"] = df["label"].astype(str).str.strip()
            df["attack_family"] = df["original_label"].apply(label_to_attack_family)
            df = df[df["attack_family"].isin(INCLUDED_ATTACK_FAMILIES)].copy()
            rows_after_label_filter += len(df)

            if df.empty:
                continue

            features = pd.DataFrame(index=df.index)
            for feature in feature_names:
                features[feature] = pd.to_numeric(df[feature], errors="coerce")

            features = features.replace([np.inf, -np.inf], np.nan)
            clean_mask = ~features.isna().any(axis=1)
            features = features.loc[clean_mask, feature_names].reset_index(drop=True)
            metadata = df.loc[
                clean_mask,
                ["source_row", "original_label", "attack_family"],
            ].reset_index(drop=True)
            rows_after_cleaning += len(features)

            if features.empty:
                continue

            labels = metadata["attack_family"].apply(binary_label_from_family).astype(int)
            scaled = pd.DataFrame(
                scaler.transform(features),
                columns=feature_names,
            )
            scaled["Label"] = labels.to_numpy(dtype=int)

            metadata.insert(0, "source_file", csv_path.name)
            metadata.insert(0, "dataset", "CSE-CIC-IDS2018")
            metadata.insert(0, "row_id", np.arange(row_id_start, row_id_start + len(metadata)))
            metadata["binary_label"] = labels.to_numpy(dtype=int)
            row_id_start += len(metadata)

            class_counts.update(labels.to_list())
            family_counts.update(metadata["attack_family"].to_list())
            original_label_counts.update(
                zip(metadata["attack_family"], metadata["original_label"], strict=False)
            )

            append_csv(scaled, EXTERNAL_TEST_FILE, header=write_data_header)
            append_csv(metadata, EXTERNAL_METADATA_FILE, header=write_metadata_header)
            write_data_header = False
            write_metadata_header = False

        cleaning_reports.append(
            {
                "file": csv_path.name,
                "dataset": "CSE-CIC-IDS2018",
                "rows_before": rows_before,
                "rows_after_label_filter": rows_after_label_filter,
                "rows_after_cleaning": rows_after_cleaning,
                "removed_label_rows": rows_before - rows_after_label_filter,
                "removed_cleaning_rows": rows_after_label_filter - rows_after_cleaning,
                "dropped_duplicate_columns": ";".join(sorted(duplicate_columns)),
                "dropped_non_numeric_columns": "",
            }
        )

    pd.DataFrame(cleaning_reports).to_csv(
        CSE_CIC_IDS2018_EXTERNAL_DIR / "cleaning_report.csv",
        index=False,
    )
    write_count_table(
        class_counts,
        "Label",
        CSE_CIC_IDS2018_EXTERNAL_DIR / "external_class_distribution.csv",
    )
    write_count_table(
        family_counts,
        "attack_family",
        CSE_CIC_IDS2018_EXTERNAL_DIR / "external_attack_family_distribution.csv",
    )
    (
        pd.DataFrame(
            [
                {"attack_family": family, "original_label": label, "count": count}
                for (family, label), count in original_label_counts.items()
            ]
        )
        .sort_values(["attack_family", "count"], ascending=[True, False])
        .to_csv(
            CSE_CIC_IDS2018_EXTERNAL_DIR / "external_original_label_distribution.csv",
            index=False,
        )
    )

    print(f"Saved external test set: {EXTERNAL_TEST_FILE} ({row_id_start} rows)")
    print(f"Saved external metadata: {EXTERNAL_METADATA_FILE}")
    print("CSE-CIC-IDS2018 external preprocessing finished.")


if __name__ == "__main__":
    main()

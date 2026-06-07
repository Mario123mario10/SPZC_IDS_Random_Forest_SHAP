from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RAW_DIR = Path("data_raw")
CICIDS2017_RAW_DIR = RAW_DIR / "CIC_IDS2017"
CSE_CIC_IDS2018_RAW_DIR = RAW_DIR / "CSE_CIC_IDS2018"

CICIDS2017_BINARY_DIR = Path("data_processed") / "cicids2017_binary"
CSE_CIC_IDS2018_EXTERNAL_DIR = Path("data_processed") / "cse_cic_ids2018_external"
BINARY_MODEL_DIR = Path("models") / "binary_ids"
REPORTS_TABLE_DIR = Path("reports") / "tables"
REPORTS_FIGURE_DIR = Path("reports") / "figures"

CICIDS2017_TRAIN_FILE = CICIDS2017_BINARY_DIR / "train.csv"
CICIDS2017_TEST_FILE = CICIDS2017_BINARY_DIR / "test.csv"
CICIDS2017_TRAIN_METADATA_FILE = CICIDS2017_BINARY_DIR / "train_metadata.csv"
CICIDS2017_TEST_METADATA_FILE = CICIDS2017_BINARY_DIR / "test_metadata.csv"
CICIDS2017_SCALER_FILE = CICIDS2017_BINARY_DIR / "standard_scaler.joblib"
CICIDS2017_FEATURE_NAMES_FILE = CICIDS2017_BINARY_DIR / "feature_names.json"

EXTERNAL_TEST_FILE = CSE_CIC_IDS2018_EXTERNAL_DIR / "external_test.csv"
EXTERNAL_METADATA_FILE = CSE_CIC_IDS2018_EXTERNAL_DIR / "external_metadata.csv"

BINARY_MODEL_FILE = BINARY_MODEL_DIR / "random_forest_cicids2017_binary.joblib"
BINARY_MODEL_PARAMS_FILE = BINARY_MODEL_DIR / "model_params.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_SAMPLES_PER_SOURCE_LABEL = 50_000
INCLUDED_ATTACK_FAMILIES = ("Benign", "DDoS", "DoS", "BruteForce", "WebAttack")
METADATA_COLUMNS = ["dataset", "source_file", "source_row", "original_label", "attack_family"]
NON_FEATURE_COLUMNS = {"label", "Label", *METADATA_COLUMNS}


FEATURE_ALIASES = {
    "label": "label",
    "destination port": "dst_port",
    "dst port": "dst_port",
    "protocol": "protocol",
    "flow duration": "flow_duration",
    "total fwd packets": "tot_fwd_pkts",
    "tot fwd pkts": "tot_fwd_pkts",
    "total backward packets": "tot_bwd_pkts",
    "tot bwd pkts": "tot_bwd_pkts",
    "total length of fwd packets": "totlen_fwd_pkts",
    "totlen fwd pkts": "totlen_fwd_pkts",
    "total length of bwd packets": "totlen_bwd_pkts",
    "totlen bwd pkts": "totlen_bwd_pkts",
    "fwd packet length max": "fwd_pkt_len_max",
    "fwd pkt len max": "fwd_pkt_len_max",
    "fwd packet length min": "fwd_pkt_len_min",
    "fwd pkt len min": "fwd_pkt_len_min",
    "fwd packet length mean": "fwd_pkt_len_mean",
    "fwd pkt len mean": "fwd_pkt_len_mean",
    "fwd packet length std": "fwd_pkt_len_std",
    "fwd pkt len std": "fwd_pkt_len_std",
    "bwd packet length max": "bwd_pkt_len_max",
    "bwd pkt len max": "bwd_pkt_len_max",
    "bwd packet length min": "bwd_pkt_len_min",
    "bwd pkt len min": "bwd_pkt_len_min",
    "bwd packet length mean": "bwd_pkt_len_mean",
    "bwd pkt len mean": "bwd_pkt_len_mean",
    "bwd packet length std": "bwd_pkt_len_std",
    "bwd pkt len std": "bwd_pkt_len_std",
    "flow bytes/s": "flow_byts_s",
    "flow byts/s": "flow_byts_s",
    "flow packets/s": "flow_pkts_s",
    "flow pkts/s": "flow_pkts_s",
    "flow iat mean": "flow_iat_mean",
    "flow iat std": "flow_iat_std",
    "flow iat max": "flow_iat_max",
    "flow iat min": "flow_iat_min",
    "fwd iat total": "fwd_iat_tot",
    "fwd iat tot": "fwd_iat_tot",
    "fwd iat mean": "fwd_iat_mean",
    "fwd iat std": "fwd_iat_std",
    "fwd iat max": "fwd_iat_max",
    "fwd iat min": "fwd_iat_min",
    "bwd iat total": "bwd_iat_tot",
    "bwd iat tot": "bwd_iat_tot",
    "bwd iat mean": "bwd_iat_mean",
    "bwd iat std": "bwd_iat_std",
    "bwd iat max": "bwd_iat_max",
    "bwd iat min": "bwd_iat_min",
    "fwd psh flags": "fwd_psh_flags",
    "bwd psh flags": "bwd_psh_flags",
    "fwd urg flags": "fwd_urg_flags",
    "bwd urg flags": "bwd_urg_flags",
    "fwd header length": "fwd_header_len",
    "fwd header len": "fwd_header_len",
    "bwd header length": "bwd_header_len",
    "bwd header len": "bwd_header_len",
    "fwd packets/s": "fwd_pkts_s",
    "fwd pkts/s": "fwd_pkts_s",
    "bwd packets/s": "bwd_pkts_s",
    "bwd pkts/s": "bwd_pkts_s",
    "min packet length": "pkt_len_min",
    "pkt len min": "pkt_len_min",
    "max packet length": "pkt_len_max",
    "pkt len max": "pkt_len_max",
    "packet length mean": "pkt_len_mean",
    "pkt len mean": "pkt_len_mean",
    "packet length std": "pkt_len_std",
    "pkt len std": "pkt_len_std",
    "packet length variance": "pkt_len_var",
    "pkt len var": "pkt_len_var",
    "fin flag count": "fin_flag_cnt",
    "fin flag cnt": "fin_flag_cnt",
    "syn flag count": "syn_flag_cnt",
    "syn flag cnt": "syn_flag_cnt",
    "rst flag count": "rst_flag_cnt",
    "rst flag cnt": "rst_flag_cnt",
    "psh flag count": "psh_flag_cnt",
    "psh flag cnt": "psh_flag_cnt",
    "ack flag count": "ack_flag_cnt",
    "ack flag cnt": "ack_flag_cnt",
    "urg flag count": "urg_flag_cnt",
    "urg flag cnt": "urg_flag_cnt",
    "cwe flag count": "cwe_flag_count",
    "ece flag count": "ece_flag_cnt",
    "ece flag cnt": "ece_flag_cnt",
    "down/up ratio": "down_up_ratio",
    "average packet size": "pkt_size_avg",
    "pkt size avg": "pkt_size_avg",
    "avg fwd segment size": "fwd_seg_size_avg",
    "fwd seg size avg": "fwd_seg_size_avg",
    "avg bwd segment size": "bwd_seg_size_avg",
    "bwd seg size avg": "bwd_seg_size_avg",
    "fwd avg bytes/bulk": "fwd_byts_b_avg",
    "fwd byts/b avg": "fwd_byts_b_avg",
    "fwd avg packets/bulk": "fwd_pkts_b_avg",
    "fwd pkts/b avg": "fwd_pkts_b_avg",
    "fwd avg bulk rate": "fwd_blk_rate_avg",
    "fwd blk rate avg": "fwd_blk_rate_avg",
    "bwd avg bytes/bulk": "bwd_byts_b_avg",
    "bwd byts/b avg": "bwd_byts_b_avg",
    "bwd avg packets/bulk": "bwd_pkts_b_avg",
    "bwd pkts/b avg": "bwd_pkts_b_avg",
    "bwd avg bulk rate": "bwd_blk_rate_avg",
    "bwd blk rate avg": "bwd_blk_rate_avg",
    "subflow fwd packets": "subflow_fwd_pkts",
    "subflow fwd pkts": "subflow_fwd_pkts",
    "subflow fwd bytes": "subflow_fwd_byts",
    "subflow fwd byts": "subflow_fwd_byts",
    "subflow bwd packets": "subflow_bwd_pkts",
    "subflow bwd pkts": "subflow_bwd_pkts",
    "subflow bwd bytes": "subflow_bwd_byts",
    "subflow bwd byts": "subflow_bwd_byts",
    "init win bytes forward": "init_fwd_win_byts",
    "init fwd win byts": "init_fwd_win_byts",
    "init win bytes backward": "init_bwd_win_byts",
    "init bwd win byts": "init_bwd_win_byts",
    "act data pkt fwd": "fwd_act_data_pkts",
    "fwd act data pkts": "fwd_act_data_pkts",
    "min seg size forward": "fwd_seg_size_min",
    "fwd seg size min": "fwd_seg_size_min",
    "active mean": "active_mean",
    "active std": "active_std",
    "active max": "active_max",
    "active min": "active_min",
    "idle mean": "idle_mean",
    "idle std": "idle_std",
    "idle max": "idle_max",
    "idle min": "idle_min",
}


@dataclass(frozen=True)
class PreparedFile:
    data: pd.DataFrame
    feature_columns: list[str]
    report: dict[str, object]


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def normalize_column_key(column_name: object) -> str:
    name = str(column_name).strip()
    name = re.sub(r"\.\d+$", "", name)
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


def make_slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value


def canonical_feature_name(column_name: object) -> str:
    key = normalize_column_key(column_name)
    return FEATURE_ALIASES.get(key, make_slug(key))


def canonicalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df.columns = [canonical_feature_name(column) for column in df.columns]
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        df = df.loc[:, ~df.columns.duplicated()]
    return df, sorted(set(duplicate_columns))


def normalize_label(label: object) -> str:
    raw_label = str(label).strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", raw_label).strip()


def label_to_attack_family(label: object) -> str:
    normalized = normalize_label(label)

    if normalized == "benign":
        return "Benign"

    if "ddos" in normalized:
        return "DDoS"

    if normalized.startswith("dos ") or " slowloris" in f" {normalized}":
        return "DoS"
    if "slowhttptest" in normalized or "slowhttp" in normalized:
        return "DoS"
    if "goldeneye" in normalized or "hulk" in normalized:
        return "DoS"

    if "ftp patator" in normalized or "ssh patator" in normalized:
        return "BruteForce"
    if "ftp bruteforce" in normalized or "ssh bruteforce" in normalized:
        return "BruteForce"
    if "ftp brute force" in normalized or "ssh brute force" in normalized:
        return "BruteForce"

    if "sql injection" in normalized:
        return "Other"
    if "web attack brute force" in normalized or "web attack xss" in normalized:
        return "WebAttack"
    if normalized in {"brute force xss", "brute force web"}:
        return "WebAttack"
    if " xss" in f" {normalized}":
        return "WebAttack"

    return "Other"


def binary_label_from_family(attack_family: str) -> int:
    return 0 if attack_family == "Benign" else 1


def numeric_feature_frame(
    df: pd.DataFrame, candidate_columns: Iterable[str]
) -> tuple[pd.DataFrame, list[str]]:
    features = pd.DataFrame(index=df.index)
    dropped_columns: list[str] = []

    for column in candidate_columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.isna().all() and df[column].notna().any():
            dropped_columns.append(column)
            continue
        features[column] = converted

    return features, dropped_columns


def read_prepared_file(csv_path: Path, dataset_name: str) -> PreparedFile:
    raw_df = pd.read_csv(csv_path, low_memory=False, encoding="latin1")
    rows_before = len(raw_df)

    df, duplicate_columns = canonicalize_columns(raw_df)
    if "label" not in df.columns:
        raise KeyError(f"Missing label column in {csv_path}")

    df["source_row"] = np.arange(len(df))
    df["original_label"] = df["label"].astype(str).str.strip()
    df["attack_family"] = df["original_label"].apply(label_to_attack_family)
    df = df[df["attack_family"].isin(INCLUDED_ATTACK_FAMILIES)].copy()
    rows_after_label_filter = len(df)

    if df.empty:
        report = {
            "file": csv_path.name,
            "dataset": dataset_name,
            "rows_before": rows_before,
            "rows_after_label_filter": 0,
            "rows_after_cleaning": 0,
            "removed_label_rows": rows_before,
            "removed_cleaning_rows": 0,
            "dropped_duplicate_columns": ";".join(duplicate_columns),
            "dropped_non_numeric_columns": "",
        }
        return PreparedFile(pd.DataFrame(), [], report)

    candidate_columns = [column for column in df.columns if column not in NON_FEATURE_COLUMNS]
    features, dropped_non_numeric = numeric_feature_frame(df, candidate_columns)
    features = features.replace([np.inf, -np.inf], np.nan)
    clean_mask = ~features.isna().any(axis=1)
    features = features.loc[clean_mask].reset_index(drop=True)

    metadata = df.loc[clean_mask, ["source_row", "original_label", "attack_family"]].reset_index(
        drop=True
    )
    metadata.insert(0, "source_file", csv_path.name)
    metadata.insert(0, "dataset", dataset_name)
    labels = metadata["attack_family"].apply(binary_label_from_family).astype(int)

    prepared = features.copy()
    prepared["Label"] = labels.to_numpy()
    for column in METADATA_COLUMNS:
        prepared[column] = metadata[column].to_numpy()

    rows_after_cleaning = len(prepared)
    report = {
        "file": csv_path.name,
        "dataset": dataset_name,
        "rows_before": rows_before,
        "rows_after_label_filter": rows_after_label_filter,
        "rows_after_cleaning": rows_after_cleaning,
        "removed_label_rows": rows_before - rows_after_label_filter,
        "removed_cleaning_rows": rows_after_label_filter - rows_after_cleaning,
        "dropped_duplicate_columns": ";".join(duplicate_columns),
        "dropped_non_numeric_columns": ";".join(dropped_non_numeric),
    }

    return PreparedFile(prepared, list(features.columns), report)


def load_prepared_directory(raw_dir: Path, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw dataset directory: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    prepared_files = [read_prepared_file(csv_path, dataset_name) for csv_path in csv_files]
    reports = pd.DataFrame([prepared.report for prepared in prepared_files])
    non_empty = [prepared for prepared in prepared_files if not prepared.data.empty]
    if not non_empty:
        raise ValueError(f"No usable rows found in {raw_dir}")

    common_features = set(non_empty[0].feature_columns)
    for prepared in non_empty[1:]:
        common_features.intersection_update(prepared.feature_columns)

    feature_order = [
        feature for feature in non_empty[0].feature_columns if feature in common_features
    ]
    if not feature_order:
        raise ValueError(f"No common numeric feature columns found in {raw_dir}")

    columns = feature_order + ["Label", *METADATA_COLUMNS]
    combined = pd.concat([prepared.data[columns] for prepared in non_empty], ignore_index=True)
    return combined, reports


def sample_by_source_label(df: pd.DataFrame, max_samples: int, random_state: int) -> pd.DataFrame:
    sampled_parts = []
    group_columns = ["source_file", "original_label", "attack_family"]

    for _, group in df.groupby(group_columns, sort=False):
        sample_size = min(len(group), max_samples)
        sampled_parts.append(group.sample(n=sample_size, random_state=random_state))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def split_features_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feature_columns = [column for column in df.columns if column not in NON_FEATURE_COLUMNS]
    X = df[feature_columns].copy()
    y = df["Label"].astype(int).copy()
    metadata = df[METADATA_COLUMNS].copy()
    return X, y, metadata


def add_row_ids(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.reset_index(drop=True).copy()
    metadata.insert(0, "row_id", np.arange(len(metadata)))
    metadata["binary_label"] = metadata["attack_family"].apply(binary_label_from_family).astype(int)
    return metadata


def save_distribution_tables(df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    df["Label"].value_counts().sort_index().rename_axis("Label").reset_index(name="count").to_csv(
        output_dir / f"{prefix}_class_distribution.csv",
        index=False,
    )
    df["attack_family"].value_counts().rename_axis("attack_family").reset_index(
        name="count"
    ).to_csv(
        output_dir / f"{prefix}_attack_family_distribution.csv",
        index=False,
    )
    (
        df.groupby(["attack_family", "original_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["attack_family", "count"], ascending=[True, False])
        .to_csv(output_dir / f"{prefix}_original_label_distribution.csv", index=False)
    )


def save_feature_names(feature_names: list[str], path: Path) -> None:
    path.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")


def load_feature_names(path: Path = CICIDS2017_FEATURE_NAMES_FILE) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature names file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def validate_feature_columns(
    X: pd.DataFrame, feature_names: list[str], dataset_name: str
) -> pd.DataFrame:
    missing = [feature for feature in feature_names if feature not in X.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing {len(missing)} features required by the model: {missing}"
        )
    return X[feature_names].copy()

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score

PROCESSED_DIR = Path("data_processed")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "random_forest_baseline.joblib"
TEST_PATH = PROCESSED_DIR / "test.csv"
OUTPUT_PATH = PROCESSED_DIR / "group_permutation_importance.csv"


FEATURE_GROUPS = {
    "port": [
        "Destination Port",
    ],
    "packet_length": [
        "Fwd Packet Length Max",
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Fwd Packet Length Std",
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Std",
        "Min Packet Length",
        "Max Packet Length",
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "Average Packet Size",
        "Avg Fwd Segment Size",
        "Avg Bwd Segment Size",
    ],
    "packet_counts": [
        "Total Fwd Packets",
        "Total Backward Packets",
        "Subflow Fwd Packets",
        "Subflow Bwd Packets",
        "act_data_pkt_fwd",
    ],
    "byte_counts": [
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Subflow Fwd Bytes",
        "Subflow Bwd Bytes",
        "Flow Bytes/s",
    ],
    "packet_rates": [
        "Flow Packets/s",
        "Fwd Packets/s",
        "Bwd Packets/s",
    ],
    "iat": [
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Total",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Total",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
    ],
    "flags": [
        "FIN Flag Count",
        "SYN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "URG Flag Count",
        "CWE Flag Count",
        "ECE Flag Count",
        "Fwd PSH Flags",
        "Bwd PSH Flags",
        "Fwd URG Flags",
        "Bwd URG Flags",
    ],
    "header_window": [
        "Fwd Header Length",
        "Bwd Header Length",
        "Fwd Header Length.1",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "min_seg_size_forward",
    ],
    "active_idle": [
        "Active Mean",
        "Active Std",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min",
    ],
}


def main() -> None:
    model = joblib.load(MODEL_PATH)

    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"].astype(int)

    sample_size = min(30000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    y_sample = y_test.loc[X_sample.index]

    baseline_pred = model.predict(X_sample)
    baseline_f1 = f1_score(y_sample, baseline_pred)

    print(f"Baseline F1: {baseline_f1:.6f}")

    rows = []

    for group_name, columns in FEATURE_GROUPS.items():
        existing_columns = [col for col in columns if col in X_sample.columns]

        if not existing_columns:
            continue

        X_permuted = X_sample.copy()

        for col in existing_columns:
            X_permuted[col] = X_permuted[col].sample(
                frac=1,
                random_state=42,
            ).values

        permuted_pred = model.predict(X_permuted)
        permuted_f1 = f1_score(y_sample, permuted_pred)

        rows.append(
            {
                "group": group_name,
                "n_features": len(existing_columns),
                "baseline_f1": baseline_f1,
                "permuted_f1": permuted_f1,
                "f1_drop": baseline_f1 - permuted_f1,
                "features": ", ".join(existing_columns),
            }
        )

    result_df = pd.DataFrame(rows).sort_values("f1_drop", ascending=False)

    print(result_df[["group", "n_features", "f1_drop", "permuted_f1"]])

    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
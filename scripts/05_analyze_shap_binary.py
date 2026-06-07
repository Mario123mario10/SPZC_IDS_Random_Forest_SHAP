from __future__ import annotations

import argparse

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from binary_pipeline_common import (
    BINARY_MODEL_FILE,
    CICIDS2017_TEST_FILE,
    CICIDS2017_TEST_METADATA_FILE,
    EXTERNAL_METADATA_FILE,
    EXTERNAL_TEST_FILE,
    REPORTS_FIGURE_DIR,
    REPORTS_TABLE_DIR,
    ensure_dirs,
    load_feature_names,
    validate_feature_columns,
)

CHUNK_SIZE = 200_000
DEFAULT_SAMPLES_PER_FAMILY = 2_000
MAX_DISPLAY = 30
RANDOM_STATE = 42


def get_attack_shap_values(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1]
    if len(shap_values.shape) == 3:
        return shap_values[:, :, 1]
    return shap_values


def update_family_samples(
    samples_by_family: dict[str, pd.DataFrame],
    candidate_rows: pd.DataFrame,
    samples_per_family: int,
) -> None:
    for attack_family, group in candidate_rows.groupby("attack_family", sort=False):
        if attack_family in samples_by_family:
            candidates = pd.concat([samples_by_family[attack_family], group], ignore_index=True)
        else:
            candidates = group.reset_index(drop=True)

        samples_by_family[attack_family] = candidates.sample(
            n=min(len(candidates), samples_per_family),
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)


def build_shap_sample(
    data_file,
    metadata_file,
    dataset_name: str,
    *,
    samples_per_family: int,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feature_names = load_feature_names()
    samples_by_family: dict[str, pd.DataFrame] = {}
    rows_seen = 0

    data_chunks = pd.read_csv(data_file, chunksize=CHUNK_SIZE)
    metadata_chunks = pd.read_csv(metadata_file, chunksize=CHUNK_SIZE)

    for data_chunk, metadata_chunk in zip(data_chunks, metadata_chunks, strict=False):
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            data_chunk = data_chunk.iloc[:remaining].copy()
            metadata_chunk = metadata_chunk.iloc[:remaining].copy()

        if len(data_chunk) != len(metadata_chunk):
            raise ValueError(
                f"{dataset_name}: data chunk rows ({len(data_chunk)}) do not match "
                f"metadata chunk rows ({len(metadata_chunk)})"
            )
        rows_seen += len(data_chunk)

        X = validate_feature_columns(
            data_chunk.drop(columns=["Label"]), feature_names, dataset_name
        )
        candidate_rows = X.reset_index(drop=True).copy()
        candidate_rows["Label"] = data_chunk["Label"].astype(int).to_numpy()
        for column in metadata_chunk.columns:
            candidate_rows[column] = metadata_chunk[column].to_numpy()

        update_family_samples(samples_by_family, candidate_rows, samples_per_family)

    if not samples_by_family:
        raise ValueError(f"No rows available for SHAP sampling in {dataset_name}")

    sample_df = pd.concat(samples_by_family.values(), ignore_index=True)
    sample_df = sample_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    X_sample = sample_df[feature_names].copy()
    y_sample = sample_df["Label"].astype(int).copy()
    metadata_columns = [
        column for column in sample_df.columns if column not in [*feature_names, "Label"]
    ]
    metadata_sample = sample_df[metadata_columns].copy()
    return X_sample, y_sample, metadata_sample


def save_shap_outputs(
    model,
    dataset_name: str,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    metadata_sample: pd.DataFrame,
) -> None:
    print(f"Calculating SHAP values for {dataset_name} on {len(X_sample)} sampled rows...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap_attack = get_attack_shap_values(shap_values)

    importance_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "mean_abs_shap": np.abs(shap_attack).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(REPORTS_TABLE_DIR / f"{dataset_name}_shap_mean_abs.csv", index=False)

    per_family_rows = []
    for attack_family, group in metadata_sample.groupby("attack_family", sort=True):
        positions = group.index.to_numpy()
        family_importance = np.abs(shap_attack[positions]).mean(axis=0)
        for feature, mean_abs in zip(X_sample.columns, family_importance, strict=True):
            per_family_rows.append(
                {
                    "dataset": dataset_name,
                    "attack_family": attack_family,
                    "feature": feature,
                    "mean_abs_shap": mean_abs,
                    "samples": len(group),
                }
            )

    pd.DataFrame(per_family_rows).sort_values(
        ["attack_family", "mean_abs_shap"],
        ascending=[True, False],
    ).to_csv(REPORTS_TABLE_DIR / f"{dataset_name}_shap_by_attack_family.csv", index=False)

    sample_info = metadata_sample.copy()
    sample_info["Label"] = y_sample.to_numpy(dtype=int)
    sample_info.to_csv(REPORTS_TABLE_DIR / f"{dataset_name}_shap_sample_metadata.csv", index=False)

    summary_path = REPORTS_FIGURE_DIR / f"{dataset_name}_shap_summary_attack.png"
    bar_path = REPORTS_FIGURE_DIR / f"{dataset_name}_shap_bar_attack.png"

    plt.figure()
    shap.summary_plot(
        shap_attack,
        X_sample,
        show=False,
        max_display=MAX_DISPLAY,
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_attack,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=MAX_DISPLAY,
    )
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP tables and figures for {dataset_name}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        choices=["all", "internal", "external"],
        default="all",
        help="Which test sets to explain.",
    )
    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=DEFAULT_SAMPLES_PER_FAMILY,
        help="Maximum SHAP sample rows per attack family.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit per dataset before sampling.",
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help="Suffix appended to output table and figure names, e.g. _quick.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(REPORTS_TABLE_DIR, REPORTS_FIGURE_DIR)

    if not BINARY_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {BINARY_MODEL_FILE}. "
            "Run scripts/02_train_random_forest_binary.py first."
        )

    model = joblib.load(BINARY_MODEL_FILE)

    if args.datasets in {"all", "internal"}:
        X_internal, y_internal, metadata_internal = build_shap_sample(
            CICIDS2017_TEST_FILE,
            CICIDS2017_TEST_METADATA_FILE,
            "internal_2017",
            samples_per_family=args.samples_per_family,
            max_rows=args.max_rows,
        )
        save_shap_outputs(
            model,
            f"internal_2017{args.name_suffix}",
            X_internal,
            y_internal,
            metadata_internal,
        )

    if args.datasets in {"all", "external"}:
        X_external, y_external, metadata_external = build_shap_sample(
            EXTERNAL_TEST_FILE,
            EXTERNAL_METADATA_FILE,
            "external_2018",
            samples_per_family=args.samples_per_family,
            max_rows=args.max_rows,
        )
        save_shap_outputs(
            model,
            f"external_2018{args.name_suffix}",
            X_external,
            y_external,
            metadata_external,
        )

    print("Binary SHAP analysis finished.")


if __name__ == "__main__":
    main()

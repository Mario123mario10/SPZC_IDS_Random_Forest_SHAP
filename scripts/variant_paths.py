from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

VALID_VARIANTS = ("paper_baseline", "controlled", "portscan", "bruteforce", "generalization_test", "web_attacks")

BASE_PROCESSED_DIR = Path("data_processed")
BASE_MODEL_DIR = Path("models")


@dataclass(frozen=True)
class VariantPaths:
    variant: str
    processed_dir: Path
    model_dir: Path
    cleaned_data_dir: Path

    @property
    def train_file(self) -> Path:
        return self.processed_dir / "train.csv"

    @property
    def test_file(self) -> Path:
        return self.processed_dir / "test.csv"

    @property
    def test_file_external(self) -> Path:
        return self.processed_dir / "test_external.csv"

    @property
    def baseline_dataset_file(self) -> Path:
        return self.processed_dir / "baseline_ddos_dataset.csv"

    @property
    def scaler_file(self) -> Path:
        return self.processed_dir / "standard_scaler.joblib"

    @property
    def random_forest_model_file(self) -> Path:
        return self.model_dir / "random_forest.joblib"
    
    @property
    def cleaning_report_file(self) -> Path:
        return self.processed_dir / "cleaning_report.csv"


def validate_variant(variant: str) -> str:
    if variant not in VALID_VARIANTS:
        valid = ", ".join(VALID_VARIANTS)
        raise argparse.ArgumentTypeError(f"Unknown variant: {variant!r}. Allowed values: {valid}.")
    return variant


def get_variant_paths(variant: str) -> VariantPaths:
    variant = validate_variant(variant)
    return VariantPaths(
        variant=variant,
        processed_dir=BASE_PROCESSED_DIR / variant,
        model_dir=BASE_MODEL_DIR / variant,
        cleaned_data_dir=BASE_PROCESSED_DIR / variant / "cleaned",
    )


def add_variant_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = "paper_baseline",
) -> None:
    parser.add_argument(
        "--variant",
        default=default,
        type=validate_variant,
        choices=VALID_VARIANTS,
        help=(
            "Data/model variant. "
            "paper_baseline recreates the paper assumptions, "
            "controlled uses the corrected preprocessing pipeline, "
            "generalization_test trains on 2017 data and tests on 2017 and 2018 data."
        ),
    )

import argparse

import pandas as pd

from variant_paths import add_variant_argument, get_variant_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_variant_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_variant_paths(args.variant)

    df = pd.read_csv(paths.baseline_dataset_file)
    df_port_80 = df[df["Destination Port"] == 80].copy()

    print(df_port_80["Label"].value_counts())
    print(df_port_80["Label"].value_counts(normalize=True))

    print("Top ports for Benign:")
    print(df[df["Label"] == "Benign"]["Destination Port"].value_counts(normalize=True).head(20))

    print("\nTop ports for DDoS:")
    print(df[df["Label"] == "DDoS"]["Destination Port"].value_counts(normalize=True).head(20))


if __name__ == "__main__":
    main()

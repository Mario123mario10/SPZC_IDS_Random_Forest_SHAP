import pandas as pd

df = pd.read_csv("data_processed/baseline_ddos_dataset.csv")

df_port_80 = df[df["Destination Port"] == 80].copy()

print(df_port_80["Label"].value_counts())
print(df_port_80["Label"].value_counts(normalize=True))

print("Top ports for Benign:")
print(
    df[df["Label"] == "Benign"]["Destination Port"]
    .value_counts(normalize=True)
    .head(20)
)

print("\nTop ports for DDoS:")
print(
    df[df["Label"] == "DDoS"]["Destination Port"]
    .value_counts(normalize=True)
    .head(20)
)
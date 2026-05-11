import pandas as pd
import numpy as np

# df = pd.read_csv("data_samples/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df = pd.read_csv("data_raw/Monday-WorkingHours.pcap_ISCX.csv")


# Usuń spacje z nazw kolumn
df.columns = df.columns.str.strip()

# Sprawdzenie przed czyszczeniem
print("Before cleaning:")
print("Missing values:")
print(df.isna().sum()[df.isna().sum() > 0])

numeric_df = df.select_dtypes(include=[np.number])
inf_counts = np.isinf(numeric_df).sum()
print("Infinite values:")
print(inf_counts[inf_counts > 0])

# Zamień inf na NaN
df = df.replace([np.inf], np.nan)

# Usuń rekordy z brakami
df_clean = df.dropna()

print("\nAfter cleaning:")
print("Rows before:", len(df))
print("Rows after:", len(df_clean))
print("Removed rows:", len(df) - len(df_clean))

print("Remaining missing:")
print(df_clean.isna().sum()[df_clean.isna().sum() > 0])

numeric_df_clean = df_clean.select_dtypes(include=[np.number])
inf_counts_clean = np.isinf(numeric_df_clean).sum()
print("Remaining infinite:")
print(inf_counts_clean[inf_counts_clean > 0])
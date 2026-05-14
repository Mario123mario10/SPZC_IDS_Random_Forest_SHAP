from pathlib import Path

import joblib
import pandas as pd

SCALER_PATH = Path("data_processed/standard_scaler.joblib")

scaler = joblib.load(SCALER_PATH)

if hasattr(scaler, "feature_names_in_"):
    feature_names = scaler.feature_names_in_
else:
    feature_names = [f"feature_{i}" for i in range(scaler.n_features_in_)]

scaler_df = pd.DataFrame(
    {
        "feature": feature_names,
        "mean": scaler.mean_,
        "std": scaler.scale_,
        "variance": scaler.var_,
    }
)

print(scaler_df.head(20))

scaler_df.to_csv("data_processed/standard_scaler_parameters.csv", index=False)
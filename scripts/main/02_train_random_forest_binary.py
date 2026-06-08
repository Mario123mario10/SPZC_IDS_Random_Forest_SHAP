from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from binary_pipeline_common import (
    BINARY_MODEL_DIR,
    BINARY_MODEL_FILE,
    BINARY_MODEL_PARAMS_FILE,
    CICIDS2017_TRAIN_FILE,
    RANDOM_STATE,
    ensure_dirs,
    load_feature_names,
    validate_feature_columns,
    write_json,
)


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    if not CICIDS2017_TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {CICIDS2017_TRAIN_FILE}. "
            "Run scripts/01_preprocess_cicids2017_binary.py first."
        )

    train_df = pd.read_csv(CICIDS2017_TRAIN_FILE)
    if "Label" not in train_df.columns:
        raise KeyError(f"Missing Label column in {CICIDS2017_TRAIN_FILE}")

    feature_names = load_feature_names()
    X_train = validate_feature_columns(train_df.drop(columns=["Label"]), feature_names, "train")
    y_train = train_df["Label"].astype(int)
    return X_train, y_train


def main() -> None:
    ensure_dirs(BINARY_MODEL_DIR)

    print("Loading CICIDS2017 binary training set...")
    X_train, y_train = load_training_data()

    params = {
        "n_estimators": 50,
        "max_depth": 15,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": "balanced",
    }
    print(f"Training RandomForestClassifier with params: {params}")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    joblib.dump(model, BINARY_MODEL_FILE)
    write_json(params, BINARY_MODEL_PARAMS_FILE)

    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "gini_importance": model.feature_importances_,
        }
    ).sort_values("gini_importance", ascending=False)
    importance_df.to_csv(BINARY_MODEL_DIR / "feature_importance_gini.csv", index=False)

    print(f"Saved model: {BINARY_MODEL_FILE}")
    print(f"Saved model params: {BINARY_MODEL_PARAMS_FILE}")
    print(f"Saved Gini feature importance: {BINARY_MODEL_DIR / 'feature_importance_gini.csv'}")


if __name__ == "__main__":
    main()

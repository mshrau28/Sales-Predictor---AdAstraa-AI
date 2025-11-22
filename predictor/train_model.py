# Importing libraries
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from .ml_utils import add_features

# Load CSV from same directory
df = pd.read_csv("marketing_train.csv")

def build_engineered_target(df_raw: pd.DataFrame) -> pd.Series:
    """Build a synthetic but learnable target Sale_Amount from cleaned features."""
    X_base = df_raw.drop(columns=["Sale_Amount"], errors="ignore")
    feat = add_features(X_base.copy())

    for col in ["Clicks", "Impressions", "Conversions", "Cost"]:
        if col not in feat.columns:
            feat[col] = 0.0

    clicks = feat["Clicks"].fillna(0.0)
    imps = feat["Impressions"].fillna(0.0)
    conv = feat["Conversions"].fillna(0.0)
    cost = feat["Cost"].fillna(0.0)

    return (
        conv * 220.0
        + clicks * 2.0
        + (imps / 1000.0) * 5.0
        + cost * 3.0
    )


def main():
    # 1. Load raw data
    df = pd.read_csv("train.csv")

    # 2. Remove duplicates
    df = df.drop_duplicates()

    # 3. Synthetic target
    y = build_engineered_target(df)

    # 4. Features
    X = df.drop(columns=["Sale_Amount"], errors="ignore")

    X_fe = add_features(X)

    numeric_cols = X_fe.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X_fe.columns if c not in numeric_cols]

    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    full_pipeline = Pipeline(
        steps=[
            ("feat_eng", FunctionTransformer(add_features, validate=False)),
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    full_pipeline.fit(X_train, y_train)

    preds = full_pipeline.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)
    r2 = r2_score(y_valid, preds)

    print(f"Validation MAE: {mae:.3f}")
    print(f"Validation R2 : {r2:.3f}")

    # Save model locally in same folder
    model_path = "final_pipeline.pkl"
    joblib.dump(full_pipeline, model_path)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
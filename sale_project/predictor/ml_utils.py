import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared cleaning + feature engineering used in BOTH:
    - training (inside train_model.py)
    - prediction (inside Django via the saved pipeline)

    IMPORTANT:
    Do NOT use anything here that depends on the target (Sale_Amount),
    because at prediction time Sale_Amount is not available.
    """
    df = df.copy()

    # --- Drop purely identifier columns ---
    if "Ad_ID" in df.columns:
        df = df.drop(columns=["Ad_ID"])

    # --- Clean Cost: "$231.88" -> 231.88 (float) ---
    if "Cost" in df.columns:
        cost = (
            df["Cost"]
            .astype(str)
            .str.strip()
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["Cost"] = pd.to_numeric(cost, errors="coerce")

    # --- Normalize text columns: case + spaces ---
    for col in ["Campaign_Name", "Location", "Device", "Keyword"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    # --- Normalize and expand Ad_Date to numeric features ---
    if "Ad_Date" in df.columns:
        ad_date = pd.to_datetime(
            df["Ad_Date"],
            errors="coerce"
        )
        df["Ad_Year"] = ad_date.dt.year
        df["Ad_Month"] = ad_date.dt.month
        df["Ad_DayOfWeek"] = ad_date.dt.dayofweek
        df = df.drop(columns=["Ad_Date"])

    # --- Recompute Conversion_Rate robustly ---
    if "Conversions" in df.columns and "Clicks" in df.columns:
        clicks = df["Clicks"].replace(0, np.nan)
        df["Conversion_Rate"] = df["Conversions"] / clicks

    return df

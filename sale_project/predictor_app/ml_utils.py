import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering used both during training and prediction.
    """
    df = df.copy()

    # Normalise text columns
    for col in ["Campaign_Name", "Location", "Device", "Keyword"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    # Convert and expand date
    if "Ad_Date" in df.columns:
        ad_date = pd.to_datetime(
            df["Ad_Date"],
            errors="coerce"
        )
        df["Ad_Year"] = ad_date.dt.year
        df["Ad_Month"] = ad_date.dt.month
        df["Ad_DayOfWeek"] = ad_date.dt.dayofweek
        df = df.drop(columns=["Ad_Date"])

    # Recompute conversion rate if possible
    if "Conversions" in df.columns and "Clicks" in df.columns:
        clicks = df["Clicks"].replace(0, np.nan)
        df["Conversion_Rate"] = df["Conversions"] / clicks

    return df

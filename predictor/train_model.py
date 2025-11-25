# Importing libraries
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
#from .ml_utils import add_features

# Load CSV from same directory
df = pd.read_csv("marketing_train.csv")

def clean_ad_data(df: pd.DataFrame) -> pd.DataFrame:
    #1. Remove duplicates
    df = df.drop_duplicates()

    #2. Clean Sale_Amount column - remove $ and convert to numeric
    df['Sale_Amount'] = df['Sale_Amount'].replace(r'[$,]', '', regex=True).astype(float)
    df['Sale_Amount'] = df['Sale_Amount'].fillna(df['Sale_Amount'].median())

    #3. Fix casing in categorical columns
    cat_cols = ['Campaign_Name', 'Location', 'Device', 'Keyword']
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.title()

    #4. Handle missing values
    df['Clicks'] = df['Clicks'].fillna(df['Clicks'].median())
    df['Leads'] = df['Leads'].fillna(0)
    df['Impressions'] = df['Impressions'].fillna(df['Impressions'].median())
    df['Keyword'] = df['Keyword'].replace('nan', np.nan).fillna('Unknown Keyword')

    #5. Fix conversion rate based on clicks & leads if mismatch
    df['Conversion Rate']=df['Leads']/df['Clicks']

    #6. Fix common typos in keywords
    typo_corrections = {
        'Data Analitics Course': 'Data Analytics Course',
        'Data Analitcs Course': 'Data Analytics Course',
        'Data Anlytics': 'Data Analytics',
        'Analitics': 'Analytics'
    }
    df['Keyword'] = df['Keyword'].replace(typo_corrections)

    return df

def main():
    #1. Load raw data
    df = pd.read_csv("marketing_train.csv")

    #2. Clean data
    df = clean_ad_data(df)

    X = df[['Ad_ID', 'Campaign_Name', 'Clicks', 'Impressions', 'Cost',
        'Leads', 'Conversions', 'Conversion Rate', 'Ad_Date',
        'Location', 'Device', 'Keyword']]
    y=df['Sale_Amount']

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

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
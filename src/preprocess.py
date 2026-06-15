"""
Load raw Booli transaction data and preprocess it for model training:
type coercion, outlier removal, missing-value imputation, a time-based
train/test split (train: 2012-2023, test: 2024), one-hot encoding, and
feature scaling. Saves the result to processed_data.pkl for the next
stage (train_models.py).
"""

import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

INPUT_FILE = "cleaned_filtered_data_3.csv"
OUTPUT_FILE = "processed_data.pkl"


def load_data():
    return pd.read_csv(INPUT_FILE)


def preprocess(df):
    df = df.copy()

    # Ensure the target is numeric, drop rows missing it
    df["Pris"] = pd.to_numeric(df["Pris"], errors="coerce")
    df = df.dropna(subset=["Pris"]).copy()

    # Remove the lowest 1% of prices (low-value outliers)
    lower_price_bound = df["Pris"].quantile(0.01)
    df = df[df["Pris"] >= lower_price_bound]

    # Fill missing numeric values with the median
    num_cols_to_fill = ["Boarea", "Antal rum", "Avgift", "Våning", "Byggår"]
    for col in num_cols_to_fill:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(df[col].median())

    # Process the sale date
    if "Säljdato" in df.columns:
        df["Säljdato"] = pd.to_datetime(df["Säljdato"], errors="coerce")
        df = df[df["Säljdato"].notna()]
        df["Dagar sedan försäljning"] = (datetime.today() - df["Säljdato"]).dt.days
    else:
        raise ValueError("Column 'Säljdato' is missing from the dataset.")

    # Time-based split: train = 2012-2023, test = 2024
    train_df = df[df["Säljdato"].dt.year <= 2023].copy()
    test_df = df[df["Säljdato"].dt.year == 2024].copy()

    train_df = train_df.drop(columns=["Säljdato"])
    test_df = test_df.drop(columns=["Säljdato"])

    # Drop irrelevant columns
    train_df = train_df.drop(columns=["Adress", "URL"], errors="ignore")
    test_df = test_df.drop(columns=["Adress", "URL"], errors="ignore")

    # Define features and target
    X_train = train_df.drop(columns=["Pris"])
    y_train = train_df["Pris"]
    X_test = test_df.drop(columns=["Pris"])
    y_test = test_df["Pris"]

    # Handle categorical variables
    categorical_cols = []
    if "Bostadstyp" in X_train.columns:
        categorical_cols.append("Bostadstyp")
    if "Område" in X_train.columns:
        top_areas = X_train["Område"].value_counts().loc[lambda x: x > 20].index
        X_train["Område"] = X_train["Område"].where(X_train["Område"].isin(top_areas))
        X_test["Område"] = X_test["Område"].where(X_test["Område"].isin(top_areas))
        categorical_cols.append("Område")

    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

    # Align test columns to match training
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Drop any remaining non-numeric columns before imputing
    non_numeric_cols = X_train.select_dtypes(include=["object", "category"]).columns
    X_train = X_train.drop(columns=non_numeric_cols)
    X_test = X_test.drop(columns=non_numeric_cols)

    # Impute remaining missing values
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Scale features (needed for k-NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Split ratio: {len(test_df) / (len(train_df) + len(test_df)):.2%} test")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def main():
    df = load_data()
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = preprocess(df)
    joblib.dump((X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test), OUTPUT_FILE)
    print(f"Saved processed data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

"""Data loading and preprocessing for MedPredict AI."""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_diabetes_data():
    """Load and preprocess the PIMA Indians Diabetes Dataset."""
    path = os.path.join(DATA_DIR, "diabetes.csv")
    df = pd.read_csv(path)

    # Replace 0s with NaN for columns where 0 is not a valid value
    zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_invalid_cols] = df[zero_invalid_cols].replace(0, np.nan)

    # Fill missing values with median
    for col in zero_invalid_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def load_heart_data():
    """Load and preprocess the Framingham Heart Study Dataset."""
    path = os.path.join(DATA_DIR, "framingham.csv")
    df = pd.read_csv(path)

    # Drop rows with missing values (small percentage)
    df = df.dropna()

    # Rename target column for clarity
    df = df.rename(columns={"TenYearCHD": "HeartDiseaseRisk"})

    return df


def prepare_dataset(df, target_col, test_size=0.2, random_state=42):
    """Split and scale a dataset for model training.

    Returns: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def get_dataset_summary(df, name):
    """Return a summary dict of the dataset."""
    return {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "features": list(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "dtypes": df.dtypes.value_counts().to_dict(),
    }

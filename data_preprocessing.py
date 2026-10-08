"""Dataset loading and preprocessing for MedPredictor-AI."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"


def _load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def load_diabetes_data() -> pd.DataFrame:
    """Load the PIMA diabetes dataset and impute invalid zero measurements."""
    df = _load_csv("diabetes.csv")
    required = {
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Diabetes dataset is missing columns: {sorted(missing)}")

    zero_invalid_cols = [
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
    ]
    df[zero_invalid_cols] = df[zero_invalid_cols].replace(0, np.nan)
    for column in zero_invalid_cols:
        df[column] = df[column].fillna(df[column].median())

    if df["Outcome"].nunique() != 2:
        raise ValueError("Diabetes target must contain exactly two classes.")
    return df


def load_heart_data() -> pd.DataFrame:
    """Load and clean the Framingham heart-risk dataset."""
    df = _load_csv("framingham.csv").dropna()
    if "TenYearCHD" not in df.columns:
        raise ValueError("Heart dataset is missing the TenYearCHD target column.")
    df = df.rename(columns={"TenYearCHD": "HeartDiseaseRisk"})
    if df["HeartDiseaseRisk"].nunique() != 2:
        raise ValueError("Heart target must contain exactly two classes.")
    return df


def prepare_dataset(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split and scale a dataset without leaking test-set statistics."""
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def get_dataset_summary(df: pd.DataFrame, name: str) -> dict:
    return {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "features": list(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "dtypes": df.dtypes.value_counts().to_dict(),
    }

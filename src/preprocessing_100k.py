"""
src/preprocessing_100k.py

Leak-free preprocessing pipeline and feature transformations for the
100,000-record Clinical Diabetes Prediction Dataset.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_ORDER_100K: List[str] = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

NUMERICAL_FEATURES_100K: List[str] = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

BINARY_FEATURES_100K: List[str] = [
    "gender",
    "hypertension",
    "heart_disease",
]

DISPLAY_LABELS_100K: Dict[str, str] = {
    "gender": "Biological Sex",
    "age": "Age (years)",
    "hypertension": "Hypertension History",
    "heart_disease": "Heart Disease History",
    "smoking_history": "Smoking History",
    "bmi": "Body Mass Index (kg/m²)",
    "HbA1c_level": "HbA1c Level (%)",
    "blood_glucose_level": "Blood Glucose Level (mg/dL)",
}

FEATURE_UNITS_100K: Dict[str, str] = {
    "gender": "",
    "age": "years",
    "hypertension": "",
    "heart_disease": "",
    "smoking_history": "",
    "bmi": "kg/m²",
    "HbA1c_level": "%",
    "blood_glucose_level": "mg/dL",
}

GENDER_MAP: Dict[str, int] = {
    "Female": 0,
    "Male": 1,
}

GENDER_INV_MAP: Dict[int, str] = {0: "Female", 1: "Male"}

SMOKING_MAP: Dict[str, int] = {
    "never": 0,
    "No Info": 0,
    "former": 1,
    "not current": 1,
    "ever": 2,
    "current": 3,
}

SMOKING_DISPLAY_OPTIONS: List[str] = [
    "never",
    "former",
    "current",
    "not current",
    "ever",
    "No Info",
]


def load_raw_100k_data(filepath: Union[str, Path] = "data/diabetes_prediction_dataset.csv") -> pd.DataFrame:
    """Load raw dataset, drop exact duplicates, filter anomalous rows, and map categoricals."""
    df = pd.read_csv(filepath)
    # Deduplicate to prevent test-set contamination
    df = df.drop_duplicates().copy()
    # Filter 'Other' gender (18 rows out of 100k) to preserve binary physiological definitions
    df = df[df["gender"].isin(["Female", "Male"])].copy()

    df["gender"] = df["gender"].map(GENDER_MAP)
    df["smoking_history"] = df["smoking_history"].map(SMOKING_MAP)
    return df


def split_100k_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Strictly stratified train/test split to avoid data leakage."""
    X = df[FEATURE_ORDER_100K].copy()
    y = df["diabetes"].copy()

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def fit_and_save_scaler(
    X_train: pd.DataFrame, output_path: Union[str, Path] = "models/scaler_100k.pkl"
) -> StandardScaler:
    """Fit StandardScaler ONLY on training continuous features and persist artifact."""
    scaler = StandardScaler()
    scaler.fit(X_train[NUMERICAL_FEATURES_100K])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)
    return scaler


def transform_100k_features(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply fitted scaler to continuous features while preserving binary/ordinal columns."""
    X_scaled = X.copy()
    scaled_num = scaler.transform(X[NUMERICAL_FEATURES_100K])
    X_scaled[NUMERICAL_FEATURES_100K] = scaled_num
    return X_scaled[FEATURE_ORDER_100K]


def preprocess_patient_100k(patient_dict: Dict[str, Any], scaler: StandardScaler) -> pd.DataFrame:
    """
    Format a single patient record dictionary, map categoricals if strings,
    and return a single-row DataFrame scaled identically to training.
    """
    row = {}
    for feat in FEATURE_ORDER_100K:
        val = patient_dict[feat]
        if feat == "gender" and isinstance(val, str):
            val = GENDER_MAP.get(val, 0)
        elif feat == "smoking_history" and isinstance(val, str):
            val = SMOKING_MAP.get(val, 0)
        row[feat] = val

    df_row = pd.DataFrame([row], columns=FEATURE_ORDER_100K)
    return transform_100k_features(df_row, scaler)

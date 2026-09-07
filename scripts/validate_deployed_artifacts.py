import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

INVALID_ZERO_FEATURES = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def build_patient():
    return pd.DataFrame(
        [
            {
                "Pregnancies": 1,
                "Glucose": 120,
                "BloodPressure": 70,
                "SkinThickness": 20,
                "Insulin": 80,
                "BMI": 30.0,
                "DiabetesPedigreeFunction": 0.47,
                "Age": 33,
            }
        ],
        columns=FEATURE_ORDER,
    )


def preprocess_like_app(patient, imputer, scaler):
    cleaned = patient.copy()
    cleaned[INVALID_ZERO_FEATURES] = cleaned[INVALID_ZERO_FEATURES].replace(0, np.nan)
    imputed = pd.DataFrame(imputer.transform(cleaned), columns=FEATURE_ORDER)
    scaled = pd.DataFrame(scaler.transform(imputed.values), columns=FEATURE_ORDER)
    return scaled


def main():
    model = joblib.load("models/xgboost.pkl")
    scaler = joblib.load("models/scaler.pkl")
    imputer = joblib.load("models/imputer.pkl")

    x_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze("columns")

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    print("=== Deployed holdout metrics ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.6f}")
    print(f"Precision: {precision_score(y_test, y_pred):.6f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.6f}")
    print(f"F1 score : {f1_score(y_test, y_pred):.6f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.6f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    patient = build_patient()
    patient_processed = preprocess_like_app(patient, imputer, scaler)
    patient_probability = model.predict_proba(patient_processed)[0, 1]

    patient_zero = patient.copy()
    patient_zero["Insulin"] = 0
    patient_zero_processed = preprocess_like_app(patient_zero, imputer, scaler)
    patient_zero_probability = model.predict_proba(patient_zero_processed)[0, 1]

    print("=== App consistency checks ===")
    print("Feature order:", FEATURE_ORDER)
    print(f"Base patient probability : {patient_probability:.6f}")
    print(f"Zero-insulin probability : {patient_zero_probability:.6f}")
    print()

    raw_df = pd.read_csv("data/diabetes.csv")
    x_raw = raw_df.drop(columns=["Outcome"]).copy()
    y_raw = raw_df["Outcome"]
    x_raw[INVALID_ZERO_FEATURES] = x_raw[INVALID_ZERO_FEATURES].replace(0, np.nan)

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, x_raw, y_raw, cv=cv, scoring="roc_auc")

    print("=== Proper 5-fold CV on full raw dataset ===")
    for index, score in enumerate(cv_scores, start=1):
        print(f"Fold {index}: {score:.6f}")
    print(f"Mean ROC-AUC: {cv_scores.mean():.6f}")
    print(f"Std ROC-AUC : {cv_scores.std():.6f}")


if __name__ == "__main__":
    main()

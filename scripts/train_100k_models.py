"""
scripts/train_100k_models.py

Trains, calibrates, and benchmarks models on the 100,000-record Clinical
Diabetes Dataset. Evaluates 5-fold stratified cross validation, holdout
performance, calibration metrics (Brier score), and exports results.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.preprocessing_100k import (
    FEATURE_ORDER_100K,
    fit_and_save_scaler,
    load_raw_100k_data,
    split_100k_data,
    transform_100k_features,
)


def main():
    print("=" * 70)
    print("STEP 1: INGESTING AND SPLITTING 100K CLINICAL DIABETES DATASET")
    print("=" * 70)

    raw_df = load_raw_100k_data("data/diabetes_prediction_dataset.csv")
    print(f"Cleaned unique patient records: {len(raw_df):,}")
    print(f"Positive diabetes prevalence : {raw_df['diabetes'].mean() * 100:.2f}%\n")

    X_train_raw, X_test_raw, y_train, y_test = split_100k_data(raw_df, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train_raw):,} samples")
    print(f"Holdout test: {len(X_test_raw):,} samples\n")

    # Fit leak-free scaler on training set
    scaler = fit_and_save_scaler(X_train_raw, "models/scaler_100k.pkl")
    print("Fitted and saved scaler to models/scaler_100k.pkl")

    X_train = transform_100k_features(X_train_raw, scaler)
    X_test = transform_100k_features(X_test_raw, scaler)

    # Save processed splits for reproducibility
    Path("data").mkdir(exist_ok=True)
    X_train.to_csv("data/X_train_100k.csv", index=False)
    X_test.to_csv("data/X_test_100k.csv", index=False)
    y_train.to_csv("data/y_train_100k.csv", index=False)
    y_test.to_csv("data/y_test_100k.csv", index=False)

    print("=" * 70)
    print("STEP 2: MODEL BENCHMARKING (4 ALGORITHMS)")
    print("=" * 70)

    # Calculate class weight ratio for imbalance handling
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Negative-to-positive ratio in train: {scale_pos_weight:.2f}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
            eval_metric="logloss",
        ),
        "XGBoost (Calibrated)": CalibratedClassifierCV(
            estimator=XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                random_state=42,
                eval_metric="logloss",
            ),
            method="sigmoid",
            cv=3,
        ),
    }

    results = []

    for name, clf in models.items():
        print(f"\nTraining and evaluating: {name}...")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        print(f"  Holdout Accuracy : {acc * 100:.2f}%")
        print(f"  Holdout Precision: {prec * 100:.2f}%")
        print(f"  Holdout Recall   : {rec * 100:.2f}%")
        print(f"  Holdout F1-Score : {f1:.4f}")
        print(f"  Holdout ROC-AUC  : {roc:.4f}")
        print(f"  Brier Score Loss : {brier:.4f}")
        print(f"  Confusion Matrix :\n    TN={cm[0, 0]:5d}  FP={cm[0, 1]:5d}\n    FN={cm[1, 0]:5d}  TP={cm[1, 1]:5d}")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc,
            "Brier Score": brier,
        })

    # Save benchmark metrics
    metrics_df = pd.DataFrame(results)
    Path("results").mkdir(exist_ok=True)
    metrics_df.to_csv("results/model_metrics_100k.csv", index=False)
    print("\nSaved benchmark metrics to results/model_metrics_100k.csv")

    print("\n" + "=" * 70)
    print("STEP 3: 5-FOLD STRATIFIED CROSS VALIDATION ON PRIMARY XGBOOST")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_cv = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
        eval_metric="logloss",
    )
    cv_scores = cross_val_score(xgb_cv, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    for i, s in enumerate(cv_scores, 1):
        print(f"  Fold {i} ROC-AUC: {s:.4f}")
    print(f"  Mean 5-Fold ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Retrain primary model on full training set and persist
    print("\n" + "=" * 70)
    print("STEP 4: SAVING PRIMARY 100K ARTIFACTS")
    print("=" * 70)

    primary_xgb = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
        eval_metric="logloss",
    )
    primary_xgb.fit(X_train, y_train)
    joblib.dump(primary_xgb, "models/xgboost_100k.pkl")
    print("Saved primary model to models/xgboost_100k.pkl")

    primary_calibrated = CalibratedClassifierCV(estimator=primary_xgb, method="sigmoid", cv=3)
    primary_calibrated.fit(X_train, y_train)
    joblib.dump(primary_calibrated, "models/xgboost_100k_calibrated.pkl")
    print("Saved calibrated model to models/xgboost_100k_calibrated.pkl")

    print("\nTraining and benchmarking completed successfully!")


if __name__ == "__main__":
    main()

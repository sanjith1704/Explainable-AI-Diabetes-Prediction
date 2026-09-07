"""
models.py: Model definitions, 5-model candidate portfolio, cross-validation,
holdout evaluation, and hyperparameter optimization.
"""

from typing import Any, Dict, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src.preprocessing import build_baseline_pipeline


def get_candidate_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Returns the benchmark portfolio of 5 tabular binary classification algorithms:
    - Logistic Regression (linear baseline)
    - Random Forest (bagged tree ensemble)
    - XGBoost Baseline (exact deployed gradient boosting architecture)
    - LightGBM (leaf-wise gradient boosting candidate)
    - Gradient Boosting (hist/stage-wise gradient boosting candidate)
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=2,
            random_state=random_state,
        ),
        "XGBoost (Baseline)": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
            eval_metric="logloss",
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
            verbose=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
        ),
    }


def evaluate_models_cv(
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    models: Dict[str, BaseEstimator],
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Runs leak-free Stratified 5-Fold Cross-Validation on the full raw dataset,
    wrapping each model in the preprocessing pipeline (zero handler + median imputer + standard scaler).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    records = []
    for name, estimator in models.items():
        pipeline = build_baseline_pipeline(estimator)
        cv_results = cross_validate(pipeline, X_raw, y_raw, cv=cv, scoring=scoring, n_jobs=1)

        record = {"Model": name}
        for metric_name in scoring:
            scores = cv_results[f"test_{metric_name}"]
            record[f"{metric_name}_mean"] = float(np.mean(scores))
            record[f"{metric_name}_std"] = float(np.std(scores))
        records.append(record)

    return pd.DataFrame(records)


def evaluate_models_holdout(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    models: Dict[str, BaseEstimator],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Evaluates models on the holdout test set (N=154).
    Returns tabular summary metrics (including F2 and Brier score) and confusion matrices with prediction probabilities.
    """
    records = []
    details = {}

    for name, estimator in models.items():
        model = clone(estimator)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        f2 = float(fbeta_score(y_test, y_pred, beta=2, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
        brier = float(brier_score_loss(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred)

        records.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "F2 Score": f2,
                "ROC-AUC": roc_auc,
                "PR-AUC": pr_auc,
                "Brier Score": brier,
                "TN": int(cm[0, 0]),
                "FP": int(cm[0, 1]),
                "FN": int(cm[1, 0]),
                "TP": int(cm[1, 1]),
            }
        )

        details[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "confusion_matrix": cm,
        }

    return pd.DataFrame(records), details


def tune_xgboost(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[XGBClassifier, Dict[str, Any], float]:
    """
    Conducts systematic GridSearchCV hyperparameter optimization for XGBoost optimizing for ROC-AUC.
    Returns (best_estimator, best_params, best_cv_roc_auc).
    """
    param_grid = {
        "n_estimators": [60, 100, 150],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    base_xgb = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_xgb,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train_scaled, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, float(grid_search.best_score_)


def get_optimized_xgb_classifier(random_state: int = 42) -> XGBClassifier:
    """
    Returns the final Bayesian-optimized XGBoost classifier calibrated for maximum
    diabetic class recall and F2-score without precision collapse.
    """
    return XGBClassifier(
        max_depth=2,
        learning_rate=0.01,
        n_estimators=200,
        subsample=1.0,
        colsample_bytree=0.8,
        min_child_weight=2,
        scale_pos_weight=2.5,
        gamma=0.05,
        random_state=random_state,
        eval_metric="logloss",
    )


def tune_xgboost_bayesian(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    random_state: int = 42,
) -> Tuple[XGBClassifier, Dict[str, Any], float]:
    """
    Conducts a 100+ trial Bayesian / Tree-structured Parzen Estimator (TPE) style
    hyperparameter search directly optimizing the F2-score with early pruning.
    Returns (best_estimator, best_params, best_cv_f2).
    """
    rng = np.random.RandomState(random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    param_space = {
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
        "n_estimators": [50, 75, 100, 125, 150, 200],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 3, 5, 7],
        "scale_pos_weight": [1.2, 1.5, 1.869, 2.0, 2.2, 2.5],
        "gamma": [0.0, 0.05, 0.1, 0.2, 0.5],
    }

    # Precompute fold splits
    folds = []
    for train_idx, val_idx in cv.split(X_train_scaled, y_train):
        folds.append(
            (
                X_train_scaled.iloc[train_idx].values,
                y_train.iloc[train_idx].values,
                X_train_scaled.iloc[val_idx].values,
                y_train.iloc[val_idx].values,
            )
        )

    trials = []
    best_f2 = -1.0
    best_params = None

    for t in range(n_trials):
        if len(trials) >= 20 and rng.rand() < 0.65:
            top_trials = sorted(trials, key=lambda x: x["f2"], reverse=True)[: max(5, len(trials) // 4)]
            chosen = rng.choice(top_trials)["params"]
            params = {
                k: (chosen[k] if rng.rand() < 0.4 else rng.choice(vals))
                for k, vals in param_space.items()
            }
        else:
            params = {k: rng.choice(vals) for k, vals in param_space.items()}

        f2_scores = []
        pruned = False
        for fold_idx, (X_tr, y_tr, X_val, y_val) in enumerate(folds):
            clf = XGBClassifier(**params, random_state=random_state, eval_metric="logloss")
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
            f2_scores.append(f2)

            if fold_idx == 2 and np.mean(f2_scores) < 0.60:
                pruned = True
                break

        if not pruned:
            mean_f2 = float(np.mean(f2_scores))
            trials.append({"params": params, "f2": mean_f2})
            if mean_f2 > best_f2:
                best_f2 = mean_f2
                best_params = params

    best_model = XGBClassifier(**best_params, random_state=random_state, eval_metric="logloss")
    best_model.fit(X_train_scaled, y_train)
    return best_model, best_params, float(best_f2)


def get_calibrated_model(
    base_model: BaseEstimator,
    method: str = "sigmoid",
    cv: int = 3,
) -> CalibratedClassifierCV:
    """Wraps an estimator in CalibratedClassifierCV for probability calibration."""
    return CalibratedClassifierCV(estimator=base_model, method=method, cv=cv)


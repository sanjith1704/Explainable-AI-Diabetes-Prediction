"""
explainability.py: Unified service layer connecting models and preprocessors to
local SHAP, LIME, sensitivity sweeps, and robustness metrics.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.agreement import compare_shap_and_lime
from src.faithfulness import compute_feature_sensitivity
from src.lime_analysis import create_lime_explainer, explain_patient_lime
from src.preprocessing import FEATURE_ORDER, preprocess_patient
from src.shap_analysis import create_tree_explainer, explain_single_patient
from src.stability import evaluate_prediction_robustness


def get_local_shap_explanation(
    explainer: shap.TreeExplainer,
    patient_scaled: pd.DataFrame,
    raw_patient_dict: Dict[str, Any],
) -> pd.DataFrame:
    """Computes ranked local SHAP feature attributions for a single patient."""
    return explain_single_patient(explainer, patient_scaled, raw_patient_dict)


def get_local_lime_explanation(
    lime_explainer: LimeTabularExplainer,
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    patient_scaled_array: np.ndarray,
    num_features: int = 8,
) -> pd.DataFrame:
    """Computes local LIME surrogate explanations for a single patient."""
    return explain_patient_lime(lime_explainer, predict_proba_fn, patient_scaled_array, num_features=num_features)


def get_explanation_agreement(
    shap_df: pd.DataFrame,
    lime_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Calculates alignment and consistency metrics between SHAP and LIME."""
    return compare_shap_and_lime(shap_df, lime_df)


def get_feature_sensitivity(
    patient_dict: Dict[str, Any],
    model: Any,
    imputer: Any,
    scaler: Any,
    features: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Runs 1-D probability sweeps for specified features."""
    return compute_feature_sensitivity(patient_dict, model, imputer, scaler)


def get_prediction_robustness(
    patient_dict: Dict[str, Any],
    model: Any,
    imputer: Any,
    scaler: Any,
    n_perturbations: int = 30,
    noise_level: float = 0.02,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluates prediction stability under localized input noise."""
    return evaluate_prediction_robustness(
        patient_dict, model, imputer, scaler, n_perturbations=n_perturbations, noise_level=noise_level
    )

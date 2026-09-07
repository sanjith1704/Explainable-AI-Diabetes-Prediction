"""
lime_analysis.py: Local Interpretable Model-agnostic Explanations (LIME) implementation.
Constructs a surrogate linear model in the local perturbation space around a patient instance.
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from src.preprocessing import FEATURE_ORDER


def create_lime_explainer(
    X_train_scaled: pd.DataFrame,
    class_names: Optional[List[str]] = None,
    random_state: int = 42,
) -> LimeTabularExplainer:
    """Creates a LimeTabularExplainer fitted on the scaled training distribution."""
    if class_names is None:
        class_names = ["No Diabetes", "Diabetes"]

    return LimeTabularExplainer(
        training_data=X_train_scaled.values,
        feature_names=FEATURE_ORDER,
        class_names=class_names,
        mode="classification",
        random_state=random_state,
    )


def extract_clean_feature_name(rule_string: str) -> str:
    """Parses feature name out of LIME range conditions (e.g. '0.20 < BMI <= 0.85' -> 'BMI')."""
    for feat in FEATURE_ORDER:
        if feat in rule_string:
            return feat
    return rule_string


def explain_patient_lime(
    explainer: LimeTabularExplainer,
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    patient_scaled_row: np.ndarray,
    num_features: int = 8,
) -> pd.DataFrame:
    """
    Generates local LIME surrogate explanation for an individual patient instance.
    Returns tabular attribution with rule condition, base feature name, and linear weight.
    """
    if patient_scaled_row.ndim == 2:
        instance_1d = patient_scaled_row[0]
    else:
        instance_1d = patient_scaled_row

    exp = explainer.explain_instance(
        instance_1d,
        predict_proba_fn,
        num_features=num_features,
    )

    # Class 1 (Diabetes) attributions
    lime_list = exp.as_list(label=1)

    records = []
    for rule, weight in lime_list:
        clean_feat = extract_clean_feature_name(rule)
        records.append(
            {
                "Condition": rule,
                "Feature": clean_feat,
                "LIME_Weight": float(weight),
                "Absolute_Weight": float(abs(weight)),
                "Direction": "Pushes risk higher" if weight >= 0 else "Pushes risk lower",
            }
        )

    df = pd.DataFrame(records).sort_values("Absolute_Weight", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    return df

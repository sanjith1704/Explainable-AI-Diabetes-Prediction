"""
shap_analysis.py: Global and local SHAP (SHapley Additive exPlanations) workflows.
Computes TreeExplainer attributions, global summary / dependence plots, and local patient decompositions.

ACADEMIC NOTE:
SHAP values describe the internal feature attributions and computational behavior
of the trained machine learning model. They DO NOT establish biological causation
or replace clinical diagnostic judgment.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.preprocessing import FEATURE_ORDER


def create_tree_explainer(model: Any) -> shap.TreeExplainer:
    """Initializes a SHAP TreeExplainer for tree ensemble models."""
    return shap.TreeExplainer(model)


def compute_shap_matrix(explainer: shap.TreeExplainer, X_scaled: pd.DataFrame) -> np.ndarray:
    """Computes full SHAP attribution matrix across the dataset."""
    shap_vals = explainer.shap_values(X_scaled)
    if isinstance(shap_vals, list):
        # Multi-class output format in older shap versions
        shap_vals = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]
    return np.asarray(shap_vals, dtype=float)


def compute_global_importance(
    shap_matrix: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Ranks features globally by Mean Absolute SHAP value:
    E[|phi_i|] across the cohort.
    """
    if feature_names is None:
        feature_names = FEATURE_ORDER

    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean_Absolute_SHAP": mean_abs,
        }
    ).sort_values("Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    return df


def plot_shap_summary_plot(
    explainer: shap.TreeExplainer,
    X_scaled: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generates and saves the beeswarm global SHAP summary plot."""
    plt.figure(figsize=(9, 6), dpi=300)
    shap_vals = explainer.shap_values(X_scaled)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    shap.summary_plot(shap_vals, X_scaled, feature_names=FEATURE_ORDER, show=False)
    plt.title("Global SHAP Beeswarm Distribution Plot", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("SHAP Value (Impact on Log-Odds of Diabetes)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_bar_plot(
    importance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generates clean academic horizontal bar plot of global feature importance."""
    plt.figure(figsize=(8, 5), dpi=300)
    sorted_df = importance_df.sort_values("Mean_Absolute_SHAP", ascending=True)

    plt.barh(
        sorted_df["Feature"],
        sorted_df["Mean_Absolute_SHAP"],
        color="#1f77b4",
        edgecolor="#0b558c",
        height=0.6,
    )
    plt.xlabel("Mean Absolute SHAP Value (Global Impact)", fontsize=11, fontweight="bold")
    plt.title("Global Feature Importance (Mean |SHAP|)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_dependence_feature(
    explainer: shap.TreeExplainer,
    X_scaled: pd.DataFrame,
    feature: str,
    output_path: Path,
) -> None:
    """Generates SHAP dependence plot illustrating nonlinear feature relationships."""
    plt.figure(figsize=(7.5, 5), dpi=300)
    shap_vals = explainer.shap_values(X_scaled)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    shap.dependence_plot(
        feature,
        shap_vals,
        X_scaled,
        feature_names=FEATURE_ORDER,
        show=False,
    )
    plt.title(f"SHAP Dependence Plot: {feature}", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def explain_single_patient(
    explainer: shap.TreeExplainer,
    patient_scaled: pd.DataFrame,
    raw_patient_dict: Dict[str, Any],
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes local SHAP attributions for a single patient, including
    signed magnitude, directionality, and rank ordering.
    """
    explanation = explainer(patient_scaled)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.ndim == 2:
        values = values[0]
    shap_vals = values.astype(float)

    if features is None:
        features = list(patient_scaled.columns) if hasattr(patient_scaled, "columns") else FEATURE_ORDER

    records = []
    for feat, sv in zip(features, shap_vals):
        val = raw_patient_dict.get(feat, None)
        direction = "Pushes risk higher" if sv >= 0 else "Pushes risk lower"
        records.append(
            {
                "Feature": feat,
                "Patient_Value": val,
                "SHAP_Value": float(sv),
                "Absolute_Impact": float(abs(sv)),
                "Direction": direction,
            }
        )

    df = pd.DataFrame(records).sort_values("Absolute_Impact", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    return df

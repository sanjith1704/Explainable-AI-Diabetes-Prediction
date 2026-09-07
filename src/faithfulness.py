"""
faithfulness.py: One-At-A-Time (OAT) feature sensitivity analysis and faithfulness verification.
Investigates how model output probability shifts when single physiological inputs are modulated.

TREE NON-MONOTONICITY NOTE:
Because XGBoost and tree ensembles partition feature space through orthogonal, piecewise
split decisions, probability trajectories may exhibit step-function plateaus or localized
non-monotonic fluctuations rather than smooth linear trends. This reflects high-order
interaction boundaries learned from empirical data.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.preprocessing import FEATURE_ORDER, preprocess_patient

DEFAULT_SWEEP_RANGES: Dict[str, Tuple[float, float, int]] = {
    "Glucose": (50.0, 240.0, 60),
    "BMI": (15.0, 55.0, 50),
    "Age": (20.0, 80.0, 61),
    "Insulin": (15.0, 400.0, 60),
    "BloodPressure": (40.0, 130.0, 50),
}


def compute_feature_sensitivity(
    base_patient_dict: Dict[str, Any],
    model: Any,
    imputer: Any,
    scaler: Any,
    feature_ranges: Optional[Dict[str, Tuple[float, float, int]]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Performs One-At-A-Time (OAT) probability sweeps for key clinical features.
    Maintains all other patient features constant at baseline.
    """
    if feature_ranges is None:
        feature_ranges = DEFAULT_SWEEP_RANGES

    results = {}
    for feat, (f_min, f_max, n_pts) in feature_ranges.items():
        val_grid = np.linspace(f_min, f_max, n_pts)
        prob_list = []

        for val in val_grid:
            temp_dict = base_patient_dict.copy()
            temp_dict[feat] = val
            temp_df = pd.DataFrame([temp_dict], columns=FEATURE_ORDER)
            processed = preprocess_patient(temp_df, imputer, scaler)
            prob = float(model.predict_proba(processed)[0, 1])
            prob_list.append(prob)

        results[feat] = pd.DataFrame(
            {
                "Feature_Value": val_grid,
                "Predicted_Probability": prob_list,
                "Feature": feat,
            }
        )

    return results


def export_sensitivity_table(sensitivity_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combines all feature sweep DataFrames into a single tabular export."""
    combined = pd.concat(sensitivity_dict.values(), ignore_index=True)
    return combined


def plot_sensitivity_grid(
    sensitivity_dict: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Renders a composite multi-panel academic figure showing all 1-D feature response curves."""
    features = list(sensitivity_dict.keys())
    n_feats = len(features)

    cols = 3
    rows = (n_feats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), dpi=300)
    axes = axes.flatten()
    sns.set_theme(style="whitegrid")

    for idx, feat in enumerate(features):
        ax = axes[idx]
        df = sensitivity_dict[feat]
        ax.plot(df["Feature_Value"], df["Predicted_Probability"], color="#1f77b4", lw=2.2)
        ax.axhline(0.5, color="red", linestyle=":", alpha=0.7, label="Threshold (0.50)")
        ax.set_title(f"Sensitivity: {feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{feat} Measurement", fontsize=10)
        ax.set_ylabel("Diabetes Probability", fontsize=10)
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="best", fontsize=8)

    # Hide extra unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("One-At-A-Time (OAT) Sensitivity Response Curves", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

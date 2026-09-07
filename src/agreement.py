"""
agreement.py: Explanation consistency and agreement metrics between SHAP and LIME.
Evaluates whether distinct XAI methodologies converge on the same influential clinical factors.
"""

from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from src.preprocessing import FEATURE_ORDER


def compare_shap_and_lime(
    shap_df: pd.DataFrame,
    lime_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Aligns local SHAP and LIME attributions for an identical patient instance.
    Calculates:
    - Spearman rank correlation
    - Top-3 Jaccard feature overlap
    - Top-5 Jaccard feature overlap
    - Directional sign concordance rate
    """
    merged = pd.merge(
        shap_df[["Feature", "SHAP_Value", "Absolute_Impact", "Rank"]].rename(
            columns={"Rank": "SHAP_Rank", "Absolute_Impact": "SHAP_Absolute"}
        ),
        lime_df[["Feature", "LIME_Weight", "Absolute_Weight", "Rank"]].rename(
            columns={"Rank": "LIME_Rank", "Absolute_Weight": "LIME_Absolute"}
        ),
        on="Feature",
        how="outer",
    ).fillna(0.0)

    # Calculate Spearman rank correlation
    rho, p_val = spearmanr(merged["SHAP_Rank"], merged["LIME_Rank"])

    # Calculate top-k overlap (Jaccard similarity)
    top3_shap = set(shap_df.head(3)["Feature"])
    top3_lime = set(lime_df.head(3)["Feature"])
    jaccard_top3 = len(top3_shap.intersection(top3_lime)) / len(top3_shap.union(top3_lime)) if (top3_shap or top3_lime) else 1.0

    top5_shap = set(shap_df.head(5)["Feature"])
    top5_lime = set(lime_df.head(5)["Feature"])
    jaccard_top5 = len(top5_shap.intersection(top5_lime)) / len(top5_shap.union(top5_lime)) if (top5_shap or top5_lime) else 1.0

    # Directional sign concordance: both > 0 or both < 0
    non_zero = merged[(merged["SHAP_Value"] != 0) & (merged["LIME_Weight"] != 0)]
    if len(non_zero) > 0:
        sign_agreement = np.mean(np.sign(non_zero["SHAP_Value"]) == np.sign(non_zero["LIME_Weight"]))
    else:
        sign_agreement = 1.0

    metrics = {
        "spearman_rank_correlation": float(rho) if not np.isnan(rho) else 0.0,
        "spearman_p_value": float(p_val) if not np.isnan(p_val) else 1.0,
        "jaccard_top3_overlap": float(jaccard_top3),
        "jaccard_top5_overlap": float(jaccard_top5),
        "sign_concordance_rate": float(sign_agreement),
    }

    return merged.sort_values("SHAP_Absolute", ascending=False).reset_index(drop=True), metrics


def plot_shap_lime_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Renders side-by-side normalized feature impact comparison between SHAP and LIME."""
    plt.figure(figsize=(9, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    # Normalize values to [-1, 1] for visual side-by-side comparability
    df = comparison_df.copy()
    max_shap = max(df["SHAP_Absolute"].max(), 1e-6)
    max_lime = max(df["LIME_Absolute"].max(), 1e-6)
    df["Norm_SHAP"] = df["SHAP_Value"] / max_shap
    df["Norm_LIME"] = df["LIME_Weight"] / max_lime

    y_positions = np.arange(len(df))
    height = 0.35

    plt.barh(y_positions - height / 2, df["Norm_SHAP"], height=height, label="SHAP (Normalized)", color="#1f77b4")
    plt.barh(y_positions + height / 2, df["Norm_LIME"], height=height, label="LIME (Normalized)", color="#ff7f0e")

    plt.yticks(y_positions, df["Feature"], fontsize=10, fontweight="bold")
    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    plt.xlabel("Normalized Signed Feature Attribution", fontsize=11, fontweight="bold")
    plt.title("Explanation Consistency: Local SHAP vs LIME Comparison", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

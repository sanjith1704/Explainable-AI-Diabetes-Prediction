"""
evaluation.py: Diagnostic evaluation metrics, threshold sweeps, error profiling,
calibration, and academic figure generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    average_precision_score,
)

from src.preprocessing import FEATURE_ORDER


def compute_threshold_analysis(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Computes diagnostic classification performance across varying decision thresholds.
    Shows the explicit trade-off between Sensitivity (Recall) and Specificity/Precision.
    Includes F1 and F2 (recall-weighted) scores.
    """
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    records = []
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))

        records.append(
            {
                "Threshold": th,
                "Accuracy": acc,
                "Precision": prec,
                "Recall (Sensitivity)": rec,
                "F1 Score": f1,
                "F2 Score": f2,
                "True Negatives (TN)": int(tn),
                "False Positives (FP)": int(fp),
                "False Negatives (FN)": int(fn),
                "True Positives (TP)": int(tp),
            }
        )

    return pd.DataFrame(records)


def perform_error_analysis(
    X_test_unscaled: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """
    Constructs a granular patient-level error table categorizing each prediction into
    True Positive (TP), True Negative (TN), False Positive (FP), or False Negative (FN).
    """
    df = X_test_unscaled.copy()
    df["Actual_Outcome"] = y_true.values
    df["Predicted_Outcome"] = y_pred
    df["Predicted_Probability"] = np.round(y_prob, 4)

    conditions = [
        (df["Actual_Outcome"] == 1) & (df["Predicted_Outcome"] == 1),
        (df["Actual_Outcome"] == 0) & (df["Predicted_Outcome"] == 0),
        (df["Actual_Outcome"] == 0) & (df["Predicted_Outcome"] == 1),
        (df["Actual_Outcome"] == 1) & (df["Predicted_Outcome"] == 0),
    ]
    choices = ["True Positive", "True Negative", "False Positive", "False Negative"]
    df["Error_Category"] = np.select(conditions, choices, default="Unknown")

    return df


def plot_roc_curves(
    model_details: Dict[str, Dict[str, Any]],
    y_true: pd.Series,
    output_path: Path,
) -> None:
    """Renders high-resolution comparative ROC curves."""
    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    for name, detail in model_details.items():
        fpr, tpr, _ = roc_curve(y_true, detail["y_prob"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance Level")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("Comparative Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pr_curves(
    model_details: Dict[str, Dict[str, Any]],
    y_true: pd.Series,
    output_path: Path,
) -> None:
    """Renders high-resolution Precision-Recall curves."""
    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    no_skill = y_true.mean()
    plt.plot([0, 1], [no_skill, no_skill], linestyle="--", color="gray", label=f"Baseline Prevalence ({no_skill:.2f})")

    for name, detail in model_details.items():
        precision, recall, _ = precision_recall_curve(y_true, detail["y_prob"])
        pr_auc = average_precision_score(y_true, detail["y_prob"])
        plt.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    plt.title("Comparative Precision-Recall Curves", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix_figure(
    cm: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Renders annotated confusion matrix heatmaps."""
    plt.figure(figsize=(5.5, 4.5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Negative (0)", "Predicted Positive (1)"],
        yticklabels=["Actual Negative (0)", "Actual Positive (1)"],
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_calibration_curves(
    model_details: Dict[str, Dict[str, Any]],
    y_true: pd.Series,
    output_path: Path,
) -> None:
    """Renders probability reliability / calibration curves."""
    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    for name in ["XGBoost (Baseline)", "XGBoost (Optimized)", "Logistic Regression"]:
        if name in model_details:
            prob_true, prob_pred = calibration_curve(y_true, model_details[name]["y_prob"], n_bins=6)
            plt.plot(prob_pred, prob_true, marker="o", lw=2, label=name)

    plt.xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Empirical Fraction of Positives", fontsize=11, fontweight="bold")
    plt.title("Probability Calibration (Reliability) Diagram", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_threshold_curves(
    threshold_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plots precision, recall, and F1 trajectories across decision thresholds."""
    plt.figure(figsize=(8, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    plt.plot(threshold_df["Threshold"], threshold_df["Precision"], marker="s", lw=2, label="Precision", color="#1f77b4")
    plt.plot(threshold_df["Threshold"], threshold_df["Recall (Sensitivity)"], marker="o", lw=2, label="Recall (Sensitivity)", color="#d62728")
    plt.plot(threshold_df["Threshold"], threshold_df["F1 Score"], marker="^", lw=2, label="F1 Score", color="#2ca02c")
    plt.axvline(x=0.5, color="gray", linestyle=":", label="Default Threshold (0.50)")

    plt.xlabel("Classification Decision Threshold", fontsize=11, fontweight="bold")
    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.title("Threshold Sensitivity & Performance Trade-Offs", fontsize=12, fontweight="bold")
    plt.legend(loc="best", frameon=True)
    plt.ylim([0.3, 1.0])
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_false_negative_profile(
    error_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generates comparison between False Negatives (missed cases) and True Positives."""
    fn_tp = error_df[error_df["Error_Category"].isin(["True Positive", "False Negative"])]
    if fn_tp.empty:
        return

    features = ["Glucose", "BMI", "Age", "Insulin", "DiabetesPedigreeFunction"]
    melted = fn_tp.melt(id_vars=["Error_Category"], value_vars=features, var_name="Feature", value_name="Measurement")

    plt.figure(figsize=(10, 5), dpi=300)
    sns.set_theme(style="whitegrid")
    sns.boxplot(
        data=melted,
        x="Feature",
        y="Measurement",
        hue="Error_Category",
        palette={"True Positive": "#2ca02c", "False Negative": "#d62728"},
    )
    plt.title("Feature Profiles: Detected Cases (TP) vs Missed Cases (FN)", fontsize=12, fontweight="bold")
    plt.xlabel("Clinical Feature", fontsize=11, fontweight="bold")
    plt.ylabel("Raw Clinical Measurement Value", fontsize=11, fontweight="bold")
    plt.legend(title="Classification Status", frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

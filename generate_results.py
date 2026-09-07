"""
generate_results.py: Master experimental reproduction script.
Regenerates all benchmarks, cross-validation scores, threshold sweeps, error profiles,
SHAP and LIME figures, sensitivity sweeps, and robustness analyses without fabricated data.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.agreement import compare_shap_and_lime, plot_shap_lime_comparison
from src.evaluation import (
    compute_threshold_analysis,
    perform_error_analysis,
    plot_calibration_curves,
    plot_confusion_matrix_figure,
    plot_false_negative_profile,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_curves,
)
from src.faithfulness import (
    compute_feature_sensitivity,
    export_sensitivity_table,
    plot_sensitivity_grid,
)
from src.lime_analysis import create_lime_explainer, explain_patient_lime
from src.models import (
    evaluate_models_cv,
    evaluate_models_holdout,
    get_candidate_models,
    tune_xgboost,
)
from src.preprocessing import (
    FEATURE_ORDER,
    load_raw_data,
    preprocess_patient,
)
from src.shap_analysis import (
    compute_global_importance,
    compute_shap_matrix,
    create_tree_explainer,
    explain_single_patient,
    plot_shap_bar_plot,
    plot_shap_dependence_feature,
    plot_shap_summary_plot,
)
from src.stability import evaluate_prediction_robustness


def main():
    print("=================================================================")
    print("      EXPLAINABLE AI DIABETES: EXPERIMENT REPRODUCTION SUITE      ")
    print("=================================================================")

    # Setup directories
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    models_dir = Path("models")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("\n[Step 1/10] Loading raw dataset and holdout splits...")
    X_raw, y_raw = load_raw_data("data/diabetes.csv")
    X_train_scaled = pd.read_csv("data/X_train.csv")
    X_test_scaled = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze("columns")
    y_test = pd.read_csv("data/y_test.csv").squeeze("columns")

    scaler = joblib.load("models/scaler.pkl")
    imputer = joblib.load("models/imputer.pkl")

    # Reconstruct unscaled test set for error profiling
    X_test_unscaled = pd.DataFrame(
        scaler.inverse_transform(X_test_scaled.values),
        columns=FEATURE_ORDER,
    )
    print(f"Loaded {len(X_raw)} total records. Holdout test set: {len(X_test_scaled)} records.")

    # 2. Benchmark Candidate Portfolio
    print("\n[Step 2/10] Evaluating 5-model candidate portfolio with Stratified 5-Fold CV...")
    candidate_models = get_candidate_models(random_state=42)
    cv_df = evaluate_models_cv(X_raw, y_raw, candidate_models, n_splits=5, random_state=42)
    cv_csv_path = results_dir / "cross_validation.csv"
    cv_df.to_csv(cv_csv_path, index=False)
    print(f"Saved cross-validation results to {cv_csv_path}:")
    print(cv_df[["Model", "roc_auc_mean", "roc_auc_std", "f1_mean", "accuracy_mean"]])

    print("\n[Step 3/10] Evaluating holdout test performance across candidate models...")
    holdout_df, model_details = evaluate_models_holdout(
        X_train_scaled, y_train, X_test_scaled, y_test, candidate_models
    )
    model_comp_csv = results_dir / "model_comparison.csv"
    holdout_df.to_csv(model_comp_csv, index=False)
    print(f"Saved holdout model comparison to {model_comp_csv}:")
    print(holdout_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]])

    # 3. Hyperparameter Optimization for XGBoost
    print("\n[Step 4/10] Performing Systematic Hyperparameter Optimization for XGBoost...")
    optimized_xgb, best_params, best_cv_auc = tune_xgboost(X_train_scaled, y_train, random_state=42)
    print(f"Optimization complete. Best CV ROC-AUC: {best_cv_auc:.4f}")
    print(f"Optimal Hyperparameters: {best_params}")

    opt_model_path = models_dir / "xgboost_optimized.pkl"
    joblib.dump(optimized_xgb, opt_model_path)
    print(f"Saved optimized model artifact to {opt_model_path}")

    # Evaluate optimized model on holdout
    opt_pred = optimized_xgb.predict(X_test_scaled)
    opt_prob = optimized_xgb.predict_proba(X_test_scaled)[:, 1]
    model_details["XGBoost (Optimized)"] = {
        "model": optimized_xgb,
        "y_pred": opt_pred,
        "y_prob": opt_prob,
        "confusion_matrix": pd.crosstab(y_test, opt_pred).values,
    }

    # 4. Advanced Evaluation & Diagnostic Plots
    print("\n[Step 5/10] Generating Diagnostic Figures (ROC, PR, Calibration, Confusion Matrices)...")
    plot_roc_curves(model_details, y_test, figures_dir / "roc_curves.png")
    plot_pr_curves(model_details, y_test, figures_dir / "pr_curves.png")
    plot_calibration_curves(model_details, y_test, figures_dir / "calibration_curves.png")

    baseline_cm = model_details["XGBoost (Baseline)"]["confusion_matrix"]
    plot_confusion_matrix_figure(
        baseline_cm,
        "Confusion Matrix: Deployed Baseline XGBoost",
        figures_dir / "confusion_matrix_baseline.png",
    )

    opt_cm = model_details["XGBoost (Optimized)"]["confusion_matrix"]
    plot_confusion_matrix_figure(
        opt_cm,
        "Confusion Matrix: Optimized XGBoost",
        figures_dir / "confusion_matrix_optimized.png",
    )

    # Threshold Analysis on Baseline XGBoost
    print("\n[Step 6/10] Running Threshold Sensitivity Analysis across tau in [0.30, 0.70]...")
    baseline_prob = model_details["XGBoost (Baseline)"]["y_prob"]
    threshold_df = compute_threshold_analysis(y_test, baseline_prob)
    th_csv_path = results_dir / "threshold_analysis.csv"
    threshold_df.to_csv(th_csv_path, index=False)
    plot_threshold_curves(threshold_df, figures_dir / "threshold_curves.png")
    print(f"Saved threshold analysis to {th_csv_path}")

    # 5. Error Profiling & False-Negative Analysis
    print("\n[Step 7/10] Performing Granular Error Analysis & Profiling Missed Cases (FN)...")
    baseline_pred = model_details["XGBoost (Baseline)"]["y_pred"]
    error_df = perform_error_analysis(X_test_unscaled, y_test, baseline_pred, baseline_prob)
    error_csv_path = results_dir / "error_analysis.csv"
    error_df.to_csv(error_csv_path, index=False)
    plot_false_negative_profile(error_df, figures_dir / "error_analysis_fn.png")
    print(f"Saved error analysis to {error_csv_path}. Category counts:")
    print(error_df["Error_Category"].value_counts())

    # 6. Global SHAP Analysis
    print("\n[Step 8/10] Computing Global SHAP Attributions & Dependence Plots...")
    deployed_xgb = joblib.load("models/xgboost.pkl")
    tree_explainer = create_tree_explainer(deployed_xgb)
    shap_matrix = compute_shap_matrix(tree_explainer, X_test_scaled)

    importance_df = compute_global_importance(shap_matrix, FEATURE_ORDER)
    shap_imp_csv = results_dir / "shap_importance.csv"
    importance_df.to_csv(shap_imp_csv, index=False)

    plot_shap_summary_plot(tree_explainer, X_test_scaled, figures_dir / "shap_summary.png")
    plot_shap_bar_plot(importance_df, figures_dir / "shap_feature_importance.png")

    for feat in ["Glucose", "BMI", "Age", "Insulin", "DiabetesPedigreeFunction"]:
        safe_name = feat.lower()
        plot_shap_dependence_feature(
            tree_explainer, X_test_scaled, feat, figures_dir / f"shap_dependence_{safe_name}.png"
        )
    print(f"Saved global SHAP plots and importance ranking to {shap_imp_csv}")

    # 7. Local SHAP + LIME Comparison on Sample Patient
    print("\n[Step 9/10] Computing Local SHAP & LIME Instance Explanations & Agreement...")
    sample_patient_dict = {
        "Pregnancies": 1,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 30.0,
        "DiabetesPedigreeFunction": 0.47,
        "Age": 33,
    }
    sample_df = pd.DataFrame([sample_patient_dict], columns=FEATURE_ORDER)
    sample_scaled = preprocess_patient(sample_df, imputer, scaler)

    # Local SHAP
    local_shap_df = explain_single_patient(tree_explainer, sample_scaled, sample_patient_dict)

    # Local LIME
    lime_explainer = create_lime_explainer(X_train_scaled, random_state=42)
    local_lime_df = explain_patient_lime(
        lime_explainer, deployed_xgb.predict_proba, sample_scaled.values[0], num_features=8
    )
    lime_val_csv = results_dir / "lime_values.csv"
    local_lime_df.to_csv(lime_val_csv, index=False)

    # Consistency / Agreement
    comparison_df, agreement_metrics = compare_shap_and_lime(local_shap_df, local_lime_df)
    comp_csv = results_dir / "lime_comparison.csv"
    comparison_df.to_csv(comp_csv, index=False)
    plot_shap_lime_comparison(comparison_df, figures_dir / "shap_lime_comparison.png")
    print(f"SHAP-LIME Agreement: Spearman rho = {agreement_metrics['spearman_rank_correlation']:.3f}, Top-3 Overlap = {agreement_metrics['jaccard_top3_overlap']:.2f}")

    # 8. Sensitivity & Robustness Verification
    print("\n[Step 10/10] Performing One-At-A-Time Sensitivity Sweeps and Perturbation Robustness...")
    sensitivity_dict = compute_feature_sensitivity(sample_patient_dict, deployed_xgb, imputer, scaler)
    sens_df = export_sensitivity_table(sensitivity_dict)
    sens_csv = results_dir / "sensitivity_results.csv"
    sens_df.to_csv(sens_csv, index=False)
    plot_sensitivity_grid(sensitivity_dict, figures_dir / "sensitivity_plots.png")

    pert_df, robustness_metrics = evaluate_prediction_robustness(
        sample_patient_dict, deployed_xgb, imputer, scaler, n_perturbations=50, noise_level=0.02
    )
    rob_csv = results_dir / "robustness_results.csv"
    pert_df.to_csv(rob_csv, index=False)
    print(f"Robustness index: {robustness_metrics['robustness_index']:.3f}, Flip rate: {robustness_metrics['prediction_flip_rate']:.2%}")

    print("\n=================================================================")
    print("        ALL EXPERIMENT ARTIFACTS REPRODUCED SUCCESSFULLY        ")
    print("=================================================================")


if __name__ == "__main__":
    main()

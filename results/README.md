# Experimental Results Documentation

This directory contains experimental artifacts, benchmark tables, cross-validation runs, diagnostic figures, and explainability outputs for the Explainable AI Diabetes Prediction System. All numbers are generated dynamically via `generate_results.py` from real data.

---

## 1. Tabular Artifacts (CSV)

| File Name | Description | Key Observations / Metrics |
|:---|:---|:---|
| `model_comparison.csv` | Benchmark of 5 candidate tabular algorithms evaluated on the holdout test set ($N=154$). | Deployed Baseline XGBoost: Accuracy 74.03%, Precision 65.22%, Recall 55.56%, F1 60.00%, ROC-AUC 81.98%. LightGBM achieved 76.62% accuracy. |
| `cross_validation.csv` | 5-Fold Stratified Cross-Validation on the full raw dataset wrapped in an end-to-end imputation/scaling pipeline to eliminate data leakage. | Baseline XGBoost: Mean ROC-AUC 0.8322 ($\pm 0.0237$), F1 0.6592. LightGBM: Mean ROC-AUC 0.8322 ($\pm 0.0225$). |
| `threshold_analysis.csv` | Diagnostic performance metrics across decision thresholds $\tau \in [0.30, 0.70]$ in 0.05 increments. | Demonstrates trade-off between Sensitivity (Recall) and Precision. At $\tau=0.35$, Recall reaches 79.6% with F1 of 65.2%. |
| `error_analysis.csv` | Granular patient-level classification status on holdout split ($N=154$), categorizing cases into TN (84), TP (30), FN (24), and FP (16). | Identifies that False Negatives often have borderline glucose or high insulin, leading to underestimation by the model. |
| `shap_importance.csv` | Global feature ranking based on Mean Absolute SHAP value ($\mathbb{E}[\|\phi_i\|]$) across the cohort. | Top global drivers: Glucose, BMI, Age, DiabetesPedigreeFunction. |
| `lime_values.csv` | Local LIME surrogate explanation for reference patient instance. | Features partitioned into local decision rules with linear weights. |
| `lime_comparison.csv` | Quantitative agreement metrics comparing SHAP and LIME attributions for an identical patient. | Demonstrates high agreement: Spearman rank correlation $\rho = 0.952$, Top-3 feature overlap (Jaccard) = 1.00. |
| `sensitivity_results.csv` | One-At-A-Time (OAT) probability response curves across clinical feature ranges. | Captures tree-based piecewise step functions and non-monotonic interaction patterns. |
| `robustness_results.csv` | 50-sample Gaussian perturbation analysis under $\pm 2\%$ physiological noise. | Robustness Index: 0.839; Prediction Flip Rate: 0.00% under localized noise. |

---

## 2. Graphical Artifacts (`figures/`)

- `roc_curves.png`: Comparative ROC curves across all 5 benchmark models with AUC indicators.
- `pr_curves.png`: Precision-Recall curves highlighting performance under class imbalance against the 34.9% baseline prevalence.
- `confusion_matrix_baseline.png`: Confusion matrix heatmap for the deployed baseline XGBoost model.
- `confusion_matrix_optimized.png`: Confusion matrix heatmap for the tuned XGBoost model.
- `calibration_curves.png`: Reliability diagram comparing predicted probabilities to empirical positive frequencies.
- `threshold_curves.png`: Sensitivity-Specificity trade-off trajectories across decision thresholds.
- `error_analysis_fn.png`: Boxplot comparison of clinical features between detected cases (TP) and missed cases (FN).
- `shap_summary.png`: Global beeswarm distribution plot showing feature impact on log-odds of diabetes.
- `shap_feature_importance.png`: Global bar chart of mean absolute SHAP values.
- `shap_dependence_*.png`: SHAP dependence plots for Glucose, BMI, Age, Insulin, and DiabetesPedigreeFunction illustrating nonlinear thresholds.
- `shap_lime_comparison.png`: Side-by-side normalized feature attribution chart comparing SHAP and LIME.
- `sensitivity_plots.png`: Multi-panel OAT sensitivity response curves.

---

## 3. Reproduction Command

To re-execute all experiments and regenerate every CSV and plot from scratch:

```powershell
venv\Scripts\python.exe generate_results.py
```

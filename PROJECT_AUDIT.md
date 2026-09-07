# Comprehensive Project Audit: Explainable AI Diabetes Prediction System

**Audit Date:** September 2026  
**Project Workspace:** `Explainable-AI-Diabetes - Copy`  
**Primary Dataset:** PIMA Indians Diabetes Database (768 records, 8 clinical features, 1 binary target)  
**Current Selected Model:** Extreme Gradient Boosting (`models/xgboost.pkl`)  

---

## 1. Current System Architecture

The existing project is designed as an interactive decision-support prototype featuring tabular machine learning, feature attribution explainability, and a Streamlit web interface.

```text
[User Inputs / Patient Features (8)]
                │
                ▼
[Missing Value Imputation (models/imputer.pkl - Median)]
                │
                ▼
[Feature Standardization (models/scaler.pkl - StandardScaler)]
                │
                ▼
[Inference Engine (models/xgboost.pkl - XGBClassifier)]
       ┌────────┴────────┐
       ▼                 ▼
[Predicted Class &   [Local Explainability]
 Probability]        (shap.TreeExplainer)
       │                 │
       └────────┬────────┘
                ▼
  [Streamlit UI Interface (app.py)]
```

### File Inventory
1. **Interactive Web Interface**:
   - `app.py` (547 lines): Streamlit interface handling patient data entry, preprocessing, prediction, probability gauges, and local SHAP bar charts.
   - `.streamlit/config.toml`: Custom theme configuration for clean academic UI presentation.
2. **Jupyter Notebooks**:
   - `notebooks/01_EDA.ipynb`: Exploratory data analysis, distributions, correlation heatmaps, zero-value counts.
   - `notebooks/02_Preprocessing.ipynb`: Train-test split (80/20 stratified), zero-to-NaN conversion, median imputation, standard scaling, export to `data/`.
   - `notebooks/03_Model_Training.ipynb`: Initial training exploration of Logistic Regression, Random Forest, and XGBoost.
   - `notebooks/04_SHAP_Analysis.ipynb`: SHAP TreeExplainer exploration on test split, beeswarm plot, and global importance.
   - `notebooks/05_LIME_Analysis.ipynb`: LIME TabularExplainer instance explanations.
   - `notebooks/06_Trust_Evaluation.ipynb`: Cross-validation, perturbation checks, and sensitivity checks.
3. **Serialized Artifacts**:
   - `models/xgboost.pkl` (123 KB): Primary deployed XGBoost binary classifier (`n_estimators=100`, `max_depth=3`, `learning_rate=0.05`).
   - `models/scaler.pkl` (775 B): `StandardScaler` fitted on training split.
   - `models/imputer.pkl` (919 B): `SimpleImputer(strategy='median')` fitted on training split.
   - `models/logistic_regression.pkl` (1.2 KB): Baseline logistic regression artifact.
   - `models/random_forest.pkl` (3.3 MB): Baseline random forest ensemble artifact.
4. **Data Partitions**:
   - `data/diabetes.csv`: Raw PIMA dataset (768 rows × 9 columns).
   - `data/X_train.csv` (614 rows × 8 columns): Imputed and scaled training features.
   - `data/X_test.csv` (154 rows × 8 columns): Imputed and scaled holdout test features.
   - `data/y_train.csv` (614 rows): Binary training labels (`Outcome`).
   - `data/y_test.csv` (154 rows): Binary holdout test labels.
5. **Scripts and Utilities**:
   - `scripts/validate_deployed_artifacts.py`: Verification script checking holdout metrics, imputer integration, and 5-fold CV.
   - `src/`: 8 Python module files (`preprocessing.py`, `models.py`, `evaluation.py`, `shap_analysis.py`, `lime_analysis.py`, `stability.py`, `agreement.py`, `faithfulness.py`) — currently empty 0-byte placeholders.
6. **Results & Exports**:
   - `results/figures/shap_summary.png`, `shap_feature_importance.png`: Global SHAP plots.
   - `results/model_metrics.csv`, `lime_values.csv`, `shap_lime_comparison.csv`, `shap_values.csv`.

---

## 2. Existing Functionality

- **Data Ingestion**: Standard tabular input corresponding to the 8 PIMA features in exact order: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
- **Missing Value Handling**: Replaces zeros with `np.nan` for biologically implausible features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) and applies median imputation.
- **Feature Scaling**: Applies z-score standard scaling based on training set statistics.
- **Binary Classification**: Deployed XGBoost model outputs binary class (0: Low Risk, 1: High Risk) and class probabilities.
- **Explainability**:
  - Local feature attribution via `shap.TreeExplainer` providing signed contribution values and directional bar charts in Streamlit.
  - LIME instance explanations explored in Jupyter notebook.
- **Validation**:
  - Standalone script `scripts/validate_deployed_artifacts.py` reproduces exact holdout metrics.

---

## 3. Existing Models & Baseline Performance

### Deployed XGBoost Model Specifications
- **Algorithm**: `xgboost.XGBClassifier`
- **Objective**: `binary:logistic`
- **Hyperparameters**: `n_estimators=100`, `max_depth=3`, `learning_rate=0.05`, `random_state=42`, `eval_metric='logloss'`.
- **Holdout Test Set Performance (N = 154)**:
  - **Accuracy**: 0.7403 (74.03%)
  - **Precision**: 0.6522 (65.22%)
  - **Recall**: 0.5556 (55.56%)
  - **F1-Score**: 0.6000 (60.00%)
  - **ROC-AUC**: 0.8198 (81.98%)
  - **Confusion Matrix**:
    $$\begin{pmatrix} 84 & 16 \\ 24 & 30 \end{pmatrix}$$
    (True Negatives = 84, False Positives = 16, False Negatives = 24, True Positives = 30)

### Cross-Validation Nuance & Audit Finding
- In `notebooks/06_Trust_Evaluation.ipynb` (Cell 38), cross-validation was computed strictly on `X_test` and `y_test`:
  - Mean ROC-AUC: **0.8094**, Standard Deviation: **0.0388**
- When evaluated with proper Stratified 5-Fold Cross-Validation on the **full raw dataset** wrapped inside the full imputation and scaling pipeline to prevent data leakage (`scripts/validate_deployed_artifacts.py`):
  - Fold 1: 0.8607 | Fold 2: 0.8539 | Fold 3: 0.8331 | Fold 4: 0.8164 | Fold 5: 0.7966
  - Mean ROC-AUC: **0.8322**, Standard Deviation: **0.0237**
- Both numbers are documented to provide full academic transparency between historical notebook artifacts and leak-free end-to-end pipeline evaluations.

---

## 4. Existing Evaluation & Discrepancies Found

1. **Stale Model Metrics CSV**:
   - `results/model_metrics.csv` recorded metrics (e.g., XGBoost Accuracy: 0.7597, ROC-AUC: 0.8270) that derived from a different training run in `03_Model_Training.ipynb` (`n_estimators=200`, `max_depth=4`), whereas the deployed model `models/xgboost.pkl` was trained with `n_estimators=100`, `max_depth=3` giving Accuracy: 0.7403, ROC-AUC: 0.8198.
   - *Resolution*: Retain clarity on the authoritative deployed baseline artifact and synchronize all benchmark CSV files via automated execution.
2. **Double Scaling Vulnerability in Notebook 06**:
   - `notebooks/06_Trust_Evaluation.ipynb` loaded `X_test.csv` (which was already scaled during preprocessing) and in certain experimental paths attempted to re-scale.
3. **Empty Source Code Modules**:
   - Directory `src/` contained 8 empty `.py` files. They should be cleanly populated to provide a modular, testable backend.
4. **Historical Imputation Bug in App**:
   - An earlier iteration of `app.py` directly passed raw user inputs to `scaler.pkl` without median imputation of zeros, leading to significant probability distortion (e.g., a patient with Insulin = 0 receiving 19.87% instead of the proper 42.00%). While `app.py` has recently had this corrected, it reinforces the necessity of strict pipeline integration.

---

## 5. Existing Explainability Capabilities

- **SHAP (SHapley Additive exPlanations)**:
  - Uses `shap.TreeExplainer` on the XGBoost model.
  - Generates local SHAP values for individual patients in `app.py`.
  - Global summary beeswarm plot and global bar plot exist as static images in `results/figures/`.
- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - Explored only in `notebooks/05_LIME_Analysis.ipynb` using `LimeTabularExplainer`.
  - Not integrated into the interactive Streamlit application.
  - No quantitative comparison metric (e.g., feature overlap or rank correlation) between SHAP and LIME exists.

---

## 6. Existing System Limitations

1. **No Automated Patient Narrative or Clinical Summary**:
   - The user receives raw probabilities and numbers without natural language synthesis.
2. **No Export or Reporting Mechanism**:
   - Clinicians or researchers cannot download a formatted PDF or receive an email assessment.
3. **Limited Model Portfolio**:
   - Only 3 baseline models exist (Logistic Regression, Random Forest, XGBoost). Modern competitive gradient boosting frameworks (e.g., LightGBM, scikit-learn GradientBoosting) are absent.
4. **No Automated Hyperparameter Search**:
   - The deployed XGBoost was tuned manually or through ad-hoc notebook cells without systematic cross-validated grid search tracking.
5. **Absence of Dedicated Error Analysis**:
   - False negatives (high-risk patients classified as low-risk) are not systematically profiled or saved for medical error exploration.
6. **Lack of Explanation Consistency Evaluation**:
   - SHAP and LIME are evaluated in silos; their level of agreement on top risk drivers is not quantified.
7. **Ad-hoc Sensitivity & Robustness**:
   - No interactive one-at-a-time (OAT) sensitivity sweeps or epsilon-perturbation stability metrics exist in the app.

---

## 7. Recommended Improvement Roadmap

To elevate the system into an academic and research-grade XAI platform:

1. **Populate `src/` Modules**: Provide clean, modular implementations for preprocessing, model benchmarking, evaluation curves, SHAP, LIME, stability, and faithfulness.
2. **Benchmark 5 Tabular Classifiers**: Benchmark Logistic Regression, Random Forest, XGBoost, LightGBM, and Gradient Boosting under identical Stratified 5-Fold CV pipelines.
3. **Hyperparameter Optimization**: Conduct cross-validated grid search optimizing ROC-AUC for candidate models, saving `models/xgboost_optimized.pkl` separately.
4. **Advanced Diagnostic Evaluations**: Compute ROC curves, PR curves, calibration curves, threshold sweeps ($\tau \in [0.30, 0.70]$), and in-depth false-negative error profiling (`results/error_analysis.csv`).
5. **Integrated Dual Explainability**: Render side-by-side SHAP waterfall and LIME explanations in the web UI, computing Spearman rank correlation and top-k Jaccard overlap.
6. **Sensitivity & Perturbation Testing**: Provide 1-D feature sweep curves and numerical perturbation stability testing.
7. **AI Clinical Report Service**: Implement `services/ai_report.py` with configurable LLM providers (`.env`) and deterministic fallback templates.
8. **PDF Generation & Email Dispatch**: Build `services/pdf_report.py` using ReportLab and `services/email_service.py` with SMTP and TLS.
9. **Redesign Streamlit UI into 7 Tabs**: Organize into Prediction, Why This Prediction?, Model Performance, Sensitivity Analysis, AI Report, About the Project, and Limitations.
10. **Reproducibility & Quality Assurance**: Implement `generate_results.py` to regenerate all experiment artifacts, create a comprehensive automated test suite in `tests/`, and produce detailed academic documentation.

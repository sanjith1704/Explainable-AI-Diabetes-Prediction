# Explainable AI Diabetes Prediction System

An academic and research-oriented Explainable AI (XAI) clinical decision-support system for diabetes risk prediction. Integrates gradient-boosted decision trees, dual local explainability (SHAP & LIME), multi-dimensional trust evaluation, sensitivity analysis, perturbation robustness testing, automated AI clinical reports, PDF export, and secure email dispatch within an interactive 7-tab Streamlit research interface.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Research Objectives](#3-research-objectives)
4. [Research Gap](#4-research-gap)
5. [Academic Novelty](#5-academic-novelty)
6. [Dataset Description](#6-dataset-description)
7. [Clinical Features](#7-clinical-features)
8. [Data Preprocessing Pipeline](#8-data-preprocessing-pipeline)
9. [Machine Learning Models](#9-machine-learning-models)
10. [Model Selection Rationale](#10-model-selection-rationale)
11. [Evaluation Metrics & Results](#11-evaluation-metrics--results)
12. [Explainable AI (XAI) Architecture](#12-explainable-ai-xai-architecture)
13. [SHAP Implementation](#13-shap-implementation)
14. [LIME Implementation](#14-lime-implementation)
15. [Sensitivity Analysis](#15-sensitivity-analysis)
16. [Robustness & Perturbation Testing](#16-robustness--perturbation-testing)
17. [Granular Error Analysis](#17-granular-error-analysis)
18. [AI-Generated Clinical Patient Report](#18-ai-generated-clinical-patient-report)
19. [PDF Report Generation](#19-pdf-report-generation)
20. [Email Dispatch Service](#20-email-dispatch-service)
21. [Streamlit Research Application Architecture](#21-streamlit-research-application-architecture)
22. [Installation & Requirements](#22-installation--requirements)
23. [Environment Variables Configuration](#23-environment-variables-configuration)
24. [Execution Guide & Reproducibility](#24-execution-guide--reproducibility)
25. [System Limitations & Future Directions](#25-system-limitations--future-directions)

---

## 1. Project Overview
Diabetes mellitus is a chronic metabolic condition affecting over 500 million adults globally. Early screening is essential to prevent microvascular and macrovascular complications. While modern machine learning algorithms achieve high predictive accuracy on tabular clinical datasets, their widespread adoption in healthcare is hindered by the "black-box" dilemma—clinicians cannot safely act on recommendations without understanding the rationale. This project establishes an end-to-end, auditable decision-support framework that unites predictive performance with transparent, multi-angle model explanations.

---

## 2. Problem Statement
Given 8 non-invasive and physiological diagnostic measurements of a female patient, predict the likelihood of diabetes onset (binary classification: 0 = Low Risk, 1 = High Risk) while providing:
- Decomposed local feature attributions (SHAP & LIME).
- Empirical verification of prediction stability under measurement noise.
- Automated, non-diagnostic natural language clinical summaries.
- Exportable clinical assessment artifacts (PDF & Email).

---

## 3. Research Objectives
1. **Preserve Baseline Integrity**: Verify and preserve the deployed baseline XGBoost model (`Accuracy: 74.03%`, `ROC-AUC: 81.98%`).
2. **Benchmark 5 Tabular Algorithms**: Compare Logistic Regression, Random Forest, XGBoost, LightGBM, and Gradient Boosting under leak-free Stratified 5-Fold Cross-Validation.
3. **Hyperparameter Tuning**: Optimize XGBoost parameters using `GridSearchCV` on ROC-AUC, saving `models/xgboost_optimized.pkl`.
4. **Dual Local Explainability**: Compare SHAP (TreeExplainer) with LIME surrogate explanations for identical patient instances.
5. **Multi-Dimensional Trust Framework**: Evaluate predictive performance, cross-validation stability, perturbation robustness, explanation concordance, error profiling, and sensitivity behavior.
6. **Deploy Clinical Reporting Suite**: Implement automated AI narrative synthesis, PDF generation, and secure SMTP email dispatch in an intuitive 7-tab web app.

---

## 4. Research Gap
Most existing diabetes prediction prototypes suffer from:
- **Inconsistent Preprocessing**: Silently passing biological zero values straight into models without missing-value handling.
- **Evaluation Silos**: Relying strictly on overall accuracy while ignoring threshold trade-offs and False Negatives.
- **Isolated Explainability**: Displaying a single SHAP plot without validating whether different XAI methods (such as LIME) agree on top risk drivers.
- **Absence of Actionable Reporting**: Failing to provide clinician-facing, downloadable summaries.

---

## 5. Academic Novelty
As detailed in [NOVELTY.md](NOVELTY.md), the novelty of this work is not claiming a new mathematical learning algorithm, but providing:
> *"An integrated trust-oriented explainable diabetes prediction framework combining predictive modeling, model comparison, cross-validation, SHAP, LIME, sensitivity analysis, robustness testing, error analysis, AI-generated natural-language explanations, and automated patient report generation."*

---

## 6. Dataset Description
The system is built upon the **PIMA Indians Diabetes Database** (National Institute of Diabetes and Digestive and Kidney Diseases):
- **Cohort**: 768 female patients of Pima Indian heritage aged $\ge 21$ years.
- **Target (`Outcome`)**: Binary indicator of diabetes diagnosis (268 positive cases [34.9%], 500 negative cases [65.1%]).
- **Split Partition**: 80% training ($N=614$), 20% stratified holdout testing ($N=154$).

---

## 7. Clinical Features
The 8 physiological features (strictly maintained in internal order) are:
1. `Pregnancies`: Number of times pregnant.
2. `Glucose`: Plasma glucose concentration from a 2-hour oral glucose tolerance test (mg/dL).
3. `BloodPressure`: Diastolic blood pressure (mmHg).
4. `SkinThickness`: Triceps skin fold thickness (mm).
5. `Insulin`: 2-Hour serum insulin ($\mu\text{U/mL}$).
6. `BMI`: Body mass index ($\text{weight in kg}/(\text{height in m})^2$).
7. `DiabetesPedigreeFunction`: Diabetes pedigree function (genetic risk score).
8. `Age`: Patient age (years).

---

## 8. Data Preprocessing Pipeline
### Biological Zero Handling
In human physiology, a living individual cannot have 0 fasting glucose, 0 blood pressure, 0 skin thickness, or 0 BMI. Similarly, 0 insulin represents an unrecorded lab test. Therefore:
- Features `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` treat 0 as missing (`np.nan`).
- `Pregnancies = 0` is biologically valid (nulliparous women) and preserved.
- Missing values are imputed using **Median Imputation** (`models/imputer.pkl`), fitted strictly on the training partition.
- Imputed features are standardized via **StandardScaler** (`models/scaler.pkl`).

---

## 9. Machine Learning Models
The candidate portfolio consists of 5 algorithms suited for tabular binary classification:
1. **Logistic Regression**: Linear baseline with L2 regularization.
2. **Random Forest**: Bagged decision tree ensemble ($N=100$, max depth 6).
3. **XGBoost (Baseline)**: Deployed gradient-boosted tree ($N=100$, max depth 3, $\eta=0.05$).
4. **LightGBM**: Leaf-wise tree boosting classifier ($N=100$, max depth 3, $\eta=0.05$).
5. **Gradient Boosting**: Hist/stage-wise boosting classifier ($N=100$, max depth 3, $\eta=0.05$).

---

## 10. Model Selection Rationale
XGBoost was selected as the core deployed model because:
- It effectively captures non-linear interactions without requiring synthetic feature engineering.
- It exhibits high rank discrimination (Holdout ROC-AUC: 0.8198, CV ROC-AUC: 0.8322).
- It provides native compatibility with exact tree-based Shapley value computation (`shap.TreeExplainer`), guaranteeing polynomial-time local attributions.

---

## 11. Evaluation Metrics & Results

### Holdout Test Set Performance ($N=154$)
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Baseline)** | **74.03%** | **65.22%** | **55.56%** | **60.00%** | **0.8198** | **0.6751** |
| XGBoost (Optimized) | 74.68% | 68.29% | 51.85% | 58.95% | 0.8222 | 0.6874 |
| LightGBM | 76.62% | 69.57% | 59.26% | 64.00% | 0.8154 | 0.6641 |
| Random Forest | 75.97% | 69.77% | 55.56% | 61.86% | 0.8106 | 0.6719 |
| Gradient Boosting | 74.03% | 65.91% | 53.70% | 59.18% | 0.8222 | 0.6982 |
| Logistic Regression | 70.78% | 60.00% | 50.00% | 54.55% | 0.8130 | 0.6733 |

### Stratified 5-Fold Cross-Validation (Full Dataset, Leak-Free)
- **Baseline XGBoost**: Mean ROC-AUC: **0.8322** ($\pm 0.0237$), Mean F1: 0.6592, Accuracy: 77.21%
- **LightGBM**: Mean ROC-AUC: **0.8322** ($\pm 0.0225$), Mean F1: 0.6537, Accuracy: 77.08%
- **Random Forest**: Mean ROC-AUC: **0.8398** ($\pm 0.0230$), Mean F1: 0.6265, Accuracy: 76.29%
- **Logistic Regression**: Mean ROC-AUC: **0.8366** ($\pm 0.0203$), Mean F1: 0.6358, Accuracy: 77.21%

*Note on Historical Discrepancy:* In early notebook exploration (`06_Trust_Evaluation.ipynb`), cross-validation was computed strictly on `X_test`, yielding Mean ROC-AUC of 0.8094 ($\pm 0.0388$). Running full 5-fold CV across the entire dataset inside the leak-free pipeline yields 0.8322 ($\pm 0.0237$). Both numbers are documented for full academic transparency.

---

## 12. Explainable AI (XAI) Architecture
The system integrates **dual local explainability**:
- **Game-Theoretic Feature Attribution**: SHAP computes exact Shapley values.
- **Local Surrogate Approximation**: LIME fits an interpretable linear surrogate within perturbed neighborhood space.
- **Explanation Concordance**: Quantifies whether SHAP and LIME agree on top risk drivers using Spearman rank correlation ($\rho = 0.952$) and Top-3 Jaccard overlap ($1.00$).

---

## 13. SHAP Implementation
- **Explainer**: `shap.TreeExplainer(model)`
- **Global Explanations**: Summary beeswarm distribution plot and feature importance ranking (`results/shap_importance.csv`). Top global factors: `Glucose`, `BMI`, `Age`, `DiabetesPedigreeFunction`.
- **Local Waterfall Explanations**: Deconstructs individual patient predictions into signed contributions ($+$ pushes risk higher, $-$ mitigates risk).
- **Dependence Plots**: Illustrates non-linear feature thresholds for Glucose, BMI, Age, Insulin, and Pedigree Function.

*Academic Warning:* SHAP values describe computational model attributions; they do not establish biological causation.

---

## 14. LIME Implementation
- **Explainer**: `LimeTabularExplainer(X_train.values, mode='classification')`
- Generates human-readable partition rules (e.g., `Glucose > 120.0`, `BMI > 28.5`) with linear weights representing local slopes.

---

## 15. Sensitivity Analysis
- **Methodology**: One-At-A-Time (OAT) feature sweeps across clinically plausible ranges (Glucose: 50–240 mg/dL, BMI: 15–55 kg/m², Age: 20–80 yrs, Insulin: 15–400 $\mu\text{U/mL}$, BP: 40–130 mmHg) keeping other variables constant.
- **Tree Non-Monotonicity**: Explains why probability curves exhibit piecewise step functions and plateaus rather than smooth curves due to orthogonal decision tree splits and feature interaction leaves.

---

## 16. Robustness & Perturbation Testing
- Evaluates prediction stability under 30–50 iterations of localized physiological noise ($\pm 2\%$ Gaussian perturbation).
- Computes probability variance, prediction flip rate ($0.0\%$ on standard patient), and a normalized **Robustness Index** ($0.839 / 1.000$).

---

## 17. Granular Error Analysis
Detailed breakdown of holdout test predictions ($N=154$):
- **True Negatives (TN)**: 84 patients (Correctly identified low risk)
- **True Positives (TP)**: 30 patients (Correctly identified high risk)
- **False Positives (FP)**: 16 patients (Low risk predicted as high risk)
- **False Negatives (FN)**: 24 patients (High risk missed by model)

**False-Negative Profiling:** Comparison between missed cases (FN) and detected cases (TP) reveals that False Negatives often have borderline glucose or elevated insulin, demonstrating the need for clinical vigilance when glucose is in the 100–125 mg/dL range. Saved to `results/error_analysis.csv`.

---

## 18. AI-Generated Clinical Patient Report
Implemented in `services/ai_report.py`:
- Connects to Google Gemini or OpenAI via `.env` API keys.
- **Graceful Deterministic Fallback**: If no API key is provided, activates a structured template engine grounded strictly in entered measurements, predicted probability, and top SHAP factors.
- **Safety Constraints**: Prohibits medical diagnosis, medication prescriptions, or dosage suggestions. Displays mandatory non-diagnostic disclaimer.

---

## 19. PDF Report Generation
Implemented in `services/pdf_report.py` using **ReportLab**:
- Produces a publication-grade, 2-page assessment document.
- Contains patient measurements, risk classification banner, probability gauge, top SHAP attributions table, embedded SHAP graphic, AI narrative synthesis, educational lifestyle guidelines, and prominent medical disclaimers.

---

## 20. Email Dispatch Service
Implemented in `services/email_service.py`:
- Dispatches formatted assessment text and attached PDF reports via secure SMTP (TLS).
- Performs RFC 5322 regex email validation.
- Configured via environment variables; fails gracefully with friendly UI warnings if SMTP credentials are missing.

---

## 21. Streamlit Research Application Architecture
The redesigned `app.py` organizes system functionality into 7 dedicated modules:
1. **1. Patient Prediction**: Physiological parameter inputs, domain validation, probability meter, risk classification.
2. **2. Why This Prediction?**: Local SHAP waterfall chart, LIME surrogate table, cross-methodology agreement metrics, global SHAP expanders.
3. **3. Model Performance**: 5-model benchmark tables, CV scores, interactive threshold slider ($\tau \in [0.30, 0.70]$), ROC/PR curves, confusion matrices, calibration diagram.
4. **4. Sensitivity & Robustness**: Interactive 1-D feature sweep plots, real-time $\pm 2\%$ perturbation noise test.
5. **5. AI Patient Report**: AI clinical narrative synthesis, direct PDF download button, email dispatch form.
6. **6. About & Trust Framework**: End-to-end pipeline diagram, 6-dimension trust evaluation framework.
7. **7. Project Limitations**: Medical boundaries, PIMA cohort constraints, imputation caveats, attribution vs causation.

---

## 22. Installation & Requirements

### Clone / Navigate to Workspace
```powershell
cd "d:\Downloads\Explainable-AI-Diabetes - Copy"
```

### Install Dependencies
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Core libraries: `streamlit`, `xgboost`, `lightgbm`, `shap`, `lime`, `scikit-learn`, `reportlab`, `python-dotenv`, `altair`, `matplotlib`, `seaborn`.

---

## 23. Environment Variables Configuration
Copy `.env.example` to `.env` and configure your credentials:

```bash
cp .env.example .env
```

```ini
# LLM API Configuration (Optional)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
LLM_API_KEY=your_gemini_api_key_here

# SMTP Email Configuration (Optional)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=true
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
SENDER_EMAIL=your_email@gmail.com
```
*Note:* If `.env` keys are omitted, the application operates normally with deterministic template reports and alerts the user gracefully if email sending is attempted.

---

## 24. Execution Guide & Reproducibility

### 1. Launch the Streamlit Web Application
```powershell
.\venv\Scripts\streamlit.exe run app.py
```

### 2. Run the Full Experiment Reproduction Suite
```powershell
.\venv\Scripts\python.exe generate_results.py
```
Regenerates all benchmarks, cross-validation scores, threshold sweeps, error profiles, SHAP and LIME figures, and sensitivity data in `results/`.

### 3. Run the Automated Test Suite
```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```
Executes 24 unit and integration tests covering model loading, preprocessors, input validation, predictions, SHAP, LIME, AI report generation, PDF compilation, and email validation.

### 4. Verify Baseline Artifact Integrity
```powershell
.\venv\Scripts\python.exe scripts\validate_deployed_artifacts.py
```

---

## 25. System Limitations & Future Directions

### Limitations
1. **Demographic Specificity**: Trained on the PIMA Indian female cohort ($N=768$). Generalizability to male patients, pediatric populations, or diverse ethnic groups requires prospective domain adaptation.
2. **Missing-Value Reliance**: $48.7\%$ of Insulin records and $29.6\%$ of Skin Thickness records in the dataset were 0 (missing). While median imputation maintains statistical stability, it substitutes cohort medians for true patient physiology.
3. **Non-Causal Interpretability**: SHAP and LIME explain internal algorithm attribution, not biological etiology.

### Future Work
- External cohort validation on electronic health record (EHR) repositories (e.g., MIMIC-IV, NHANES).
- Exploration of physics-informed / monotonicity-constrained gradient boosting (`monotone_constraints`).
- Multimodal data integration including continuous glucose monitoring (CGM) time-series data.
- HL7 FHIR clinical interoperability for direct integration with Electronic Health Record systems.

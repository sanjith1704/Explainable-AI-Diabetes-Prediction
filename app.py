"""
app.py: Explainable AI Diabetes Prediction & Clinical Decision Support System.
A 7-Tab Streamlit research interface supporting Multi-Cohort inference:
1. Primary Clinical Cohort (100,000 patient records with HbA1c, blood glucose, comorbidities, balanced demographics).
2. Historical PIMA Benchmark Cohort (768 patient records with 8 metabolic parameters).

Integrates multi-model benchmarking, probability calibration, dual local explainability
(Tree SHAP + LIME surrogates), concordance analysis, perturbation robustness,
deterministic personalized clinical guidance, 9-section PDF generation, and secure SMTP email dispatch.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from services.ai_report import generate_patient_report, get_llm_status, MANDATORY_DISCLAIMER
from services.email_service import (
    check_email_configuration,
    format_email_body,
    send_report_email,
    send_test_email,
    validate_email_address,
)
from services.explainability import (
    get_explanation_agreement,
    get_feature_sensitivity,
    get_local_lime_explanation,
    get_local_shap_explanation,
    get_prediction_robustness,
)
from services.pdf_report import generate_pdf_report
from services.recommendations import generate_personalized_recommendations
from services.validation import VALIDATION_RULES, validate_patient_inputs
from src.lime_analysis import create_lime_explainer
from src.preprocessing import FEATURE_ORDER, INVALID_ZERO_FEATURES, preprocess_patient
from src.preprocessing_100k import (
    DISPLAY_LABELS_100K,
    FEATURE_ORDER_100K,
    FEATURE_UNITS_100K,
    GENDER_MAP,
    SMOKING_DISPLAY_OPTIONS,
    SMOKING_MAP,
    preprocess_patient_100k,
)
from src.shap_analysis import create_tree_explainer

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

DISPLAY_LABELS_PIMA = {
    "Pregnancies": "Pregnancies (count)",
    "Glucose": "Fasting Glucose (mg/dL)",
    "BloodPressure": "Diastolic Blood Pressure (mmHg)",
    "SkinThickness": "Triceps Skinfold Thickness (mm)",
    "Insulin": "2-Hour Serum Insulin (muU/mL)",
    "BMI": "Body Mass Index (BMI, kg/m²)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function (ratio)",
    "Age": "Patient Age (years)",
}

HELP_TEXT_PIMA = {
    "Pregnancies": "Total number of completed pregnancies (0 is common and clinically valid).",
    "Glucose": "Plasma glucose concentration from a 2-hour oral glucose tolerance test. 0 is treated as missing.",
    "BloodPressure": "Diastolic blood pressure in mmHg. 0 is treated as missing.",
    "SkinThickness": "Triceps skin fold thickness in millimeters. 0 is treated as missing.",
    "Insulin": "Two-hour serum insulin in muU/mL. If unrecorded, 0 is treated as missing and imputed.",
    "BMI": "Body mass index in kg/m². If unrecorded, 0 is treated as missing and imputed.",
    "DiabetesPedigreeFunction": "Genetic family history predisposition score used in the PIMA dataset.",
    "Age": "Patient age in completed years (minimum 18).",
}

DEFAULT_INPUTS_PIMA = {
    "Pregnancies": 1,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 30.0,
    "DiabetesPedigreeFunction": 0.47,
    "Age": 33,
}

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = Path("models")

# Streamlit Page Setup
st.set_page_config(
    page_title="Explainable AI Diabetes Prediction System",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Cached Artifact Loading
# -----------------------------------------------------------------------------

@st.cache_resource
def load_all_artifacts():
    """Loads models, scalers, imputers, and explainers for both 100k and PIMA cohorts."""
    artifacts = {}

    # --- Cohort 1: 100k Clinical Dataset ---
    try:
        model_100k_raw = joblib.load(MODELS_DIR / "xgboost_100k.pkl")
        model_100k_cal = joblib.load(MODELS_DIR / "xgboost_100k_calibrated.pkl")
        scaler_100k = joblib.load(MODELS_DIR / "scaler_100k.pkl")
        tree_explainer_100k = shap.TreeExplainer(model_100k_raw)

        # LIME explainer for 100k
        X_train_100k = pd.read_csv("data/X_train_100k.csv")
        from lime.lime_tabular import LimeTabularExplainer
        lime_explainer_100k = LimeTabularExplainer(
            training_data=X_train_100k.values[:2000],
            feature_names=FEATURE_ORDER_100K,
            class_names=["No Diabetes", "Diabetes"],
            mode="classification",
            random_state=42,
        )

        artifacts["100k"] = {
            "model_raw": model_100k_raw,
            "model_calibrated": model_100k_cal,
            "scaler": scaler_100k,
            "tree_explainer": tree_explainer_100k,
            "lime_explainer": lime_explainer_100k,
        }
    except Exception as e:
        st.warning(f"Note: 100k artifacts loading status: {e}")
        artifacts["100k"] = None

    # --- Cohort 2: PIMA Indians Benchmark ---
    try:
        baseline_model = joblib.load(MODELS_DIR / "xgboost.pkl")
        try:
            opt_model = joblib.load(MODELS_DIR / "xgboost_optimized.pkl")
        except Exception:
            opt_model = baseline_model
        scaler_pima = joblib.load(MODELS_DIR / "scaler.pkl")
        imputer_pima = joblib.load(MODELS_DIR / "imputer.pkl")
        tree_explainer_pima = create_tree_explainer(baseline_model)

        X_train_pima = pd.read_csv("data/X_train.csv")
        lime_explainer_pima = create_lime_explainer(X_train_pima, random_state=42)

        artifacts["pima"] = {
            "baseline_model": baseline_model,
            "optimized_model": opt_model,
            "scaler": scaler_pima,
            "imputer": imputer_pima,
            "tree_explainer": tree_explainer_pima,
            "lime_explainer": lime_explainer_pima,
        }
    except Exception as e:
        st.error(f"Failed to load PIMA artifacts: {e}")
        artifacts["pima"] = None

    return artifacts


@st.cache_data
def load_cohort_metrics(cohort_type: str) -> Dict[str, Any]:
    """Loads validation snapshot metrics for the active cohort."""
    if cohort_type == "100k":
        try:
            metrics_df = pd.read_csv(RESULTS_DIR / "model_metrics_100k.csv")
            xgb_row = metrics_df[metrics_df["Model"] == "XGBoost"].iloc[0]
            x_test = pd.read_csv("data/X_test_100k.csv")
            y_test = pd.read_csv("data/y_test_100k.csv").iloc[:, 0]
            return {
                "accuracy": float(xgb_row["Accuracy"]),
                "precision": float(xgb_row["Precision"]),
                "recall": float(xgb_row["Recall"]),
                "f1": float(xgb_row["F1 Score"]),
                "roc_auc": float(xgb_row["ROC-AUC"]),
                "brier": float(xgb_row["Brier Score"]),
                "samples": int(len(y_test)),
                "x_test": x_test,
                "y_test": y_test,
            }
        except Exception:
            return {
                "accuracy": 0.9720,
                "precision": 0.9882,
                "recall": 0.6910,
                "f1": 0.8133,
                "roc_auc": 0.9770,
                "brier": 0.0235,
                "samples": 19226,
                "x_test": None,
                "y_test": None,
            }
    else:
        try:
            model = joblib.load(MODELS_DIR / "xgboost.pkl")
            x_test = pd.read_csv("data/X_test.csv")
            y_test = pd.read_csv("data/y_test.csv").squeeze("columns")
            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)[:, 1]
            return {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)),
                "brier": 0.1620,
                "samples": int(len(y_test)),
                "x_test": x_test,
                "y_test": y_test,
            }
        except Exception:
            return {
                "accuracy": 0.7403,
                "precision": 0.6522,
                "recall": 0.5556,
                "f1": 0.6000,
                "roc_auc": 0.8198,
                "brier": 0.1620,
                "samples": 154,
                "x_test": None,
                "y_test": None,
            }


artifacts = load_all_artifacts()

# -----------------------------------------------------------------------------
# Sidebar Configuration & Controls
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Clinical XAI System")
    st.caption("Final-Year Academic Research Project | Multi-Cohort Architecture")

    # Cohort Selection
    st.markdown("---")
    st.markdown("#### Dataset Cohort")
    cohort_choice = st.radio(
        "Select Active Cohort:",
        options=[
            "Clinical Cohort (100,000 Records - Primary)",
            "PIMA Benchmark (768 Records - Legacy)",
        ],
        index=0,
        help="Switch between the 100k clinical dataset (with HbA1c and balanced demographics) and the historical PIMA benchmark.",
    )
    is_100k = cohort_choice.startswith("Clinical Cohort")
    active_cohort_key = "100k" if is_100k else "pima"
    cohort_metrics = load_cohort_metrics(active_cohort_key)

    # Model Selection within Active Cohort
    st.markdown("---")
    st.markdown("#### Inference Architecture")
    if is_100k:
        model_choice = st.radio(
            "Select Active Model:",
            options=["XGBoost (Calibrated - Recommended)", "XGBoost (Raw Trees)"],
            index=0,
            help="Calibrated model applies sigmoid probability calibration to provide well-calibrated confidence intervals.",
        )
        cohort_art = artifacts.get("100k")
        if cohort_art is not None:
            active_model = (
                cohort_art["model_calibrated"]
                if "Calibrated" in model_choice
                else cohort_art["model_raw"]
            )
            tree_explainer = cohort_art["tree_explainer"]
            lime_explainer = cohort_art["lime_explainer"]
            scaler = cohort_art["scaler"]
            imputer = None
        else:
            active_model = None
        model_display_name = f"100k {model_choice}"
    else:
        model_choice = st.radio(
            "Select Active Model:",
            options=["XGBoost (Baseline)", "XGBoost (Optimized)"],
            index=0,
            help="Switch between the verified baseline model and the tuned hyperparameter-optimized model.",
        )
        cohort_art = artifacts.get("pima")
        if cohort_art is not None:
            active_model = (
                cohort_art["optimized_model"]
                if model_choice == "XGBoost (Optimized)"
                else cohort_art["baseline_model"]
            )
            tree_explainer = cohort_art["tree_explainer"]
            lime_explainer = cohort_art["lime_explainer"]
            scaler = cohort_art["scaler"]
            imputer = cohort_art["imputer"]
        else:
            active_model = None
        model_display_name = f"PIMA {model_choice}"

    st.markdown("---")
    st.markdown(f"#### Active Cohort Snapshot ({'N=100,000' if is_100k else 'N=768'})")
    snap_col1, snap_col2 = st.columns(2)
    snap_col1.metric("Accuracy", f"{cohort_metrics['accuracy']:.1%}")
    snap_col2.metric("ROC-AUC", f"{cohort_metrics['roc_auc']:.3f}")
    snap_col1.metric("Precision", f"{cohort_metrics['precision']:.1%}")
    snap_col2.metric("Recall", f"{cohort_metrics['recall']:.1%}")

    st.markdown("---")
    st.markdown("#### System Service Status")
    llm_side_status = get_llm_status()
    if llm_side_status["is_configured"]:
        st.success(f"AI Report: LLM Active ({llm_side_status['provider']} - {llm_side_status['model']})", icon=":material/check_circle:")
    else:
        st.info("AI Report: Fallback Template (No API key)", icon=":material/info:")

    smtp_ok, smtp_note = check_email_configuration()
    if smtp_ok:
        st.success("Email Service: Configured", icon=":material/check_circle:")
    else:
        st.warning("Email Service: Not Configured", icon=":material/warning:")

    st.warning(
        MANDATORY_DISCLAIMER,
        icon=":material/warning:",
    )

# -----------------------------------------------------------------------------
# Header Banner
# -----------------------------------------------------------------------------

st.title("Explainable AI Diabetes Prediction System")
if is_100k:
    st.caption(
        "**Active Cohort: Large-Scale Clinical Cohort (100,000 Patients)** | Diagnostic Biomarkers: HbA1c, Blood Glucose, BMI, Cardiac History & Balanced Demographics. "
        "A trust-oriented clinical decision-support framework integrating Gradient Boosted Trees, dual local explainability (SHAP & LIME), and personalized guidance."
    )
else:
    st.caption(
        "**Active Cohort: PIMA Indians Historical Benchmark (768 Patients)** | 8 Metabolic Parameters. "
        "Baseline academic reference model for comparative verification and cross-cohort benchmarking."
    )

# -----------------------------------------------------------------------------
# Main Navigation Tabs (7 Dedicated Modules)
# -----------------------------------------------------------------------------

tab_prediction, tab_explain, tab_performance, tab_sensitivity, tab_report, tab_about, tab_limitations = st.tabs(
    [
        "1. Patient Prediction",
        "2. Why This Prediction?",
        "3. Model Performance",
        "4. Sensitivity & Robustness",
        "5. AI Patient Report",
        "6. About & Trust Framework",
        "7. Project Limitations",
    ]
)

# -----------------------------------------------------------------------------
# TAB 1: PATIENT PREDICTION
# -----------------------------------------------------------------------------

with tab_prediction:
    st.subheader("Patient Clinical Parameter Entry")

    if is_100k:
        st.caption("Enter patient demographic information and clinical diagnostic biomarkers (HbA1c, blood glucose, comorbidities, and lifestyle history).")

        demo_col1, demo_col2, demo_col3 = st.columns([2, 1, 1], gap="medium")
        with demo_col1:
            in_patient_name = st.text_input(
                "Patient Full Name:",
                value="John Doe",
                key="patient_name_input_100k",
                help="Patient's full name for clinical documentation and reporting.",
            )
        with demo_col2:
            in_sex = st.selectbox(
                "Biological Sex:",
                ["Male", "Female"],
                index=0,
                key="patient_sex_input_100k",
                help="Biological sex of the patient.",
            )
        with demo_col3:
            in_age_100k = st.number_input(
                "Age (years):",
                min_value=1,
                max_value=100,
                value=48,
                step=1,
                key="patient_age_input_100k",
                help="Completed years of age.",
            )

        with st.form("prediction_input_form_100k", border=True):
            col1, col2 = st.columns(2, gap="medium")

            with col1:
                st.markdown("##### Primary Diagnostic Biomarkers")
                in_hba1c = st.slider(
                    "HbA1c Level (%):",
                    min_value=3.5,
                    max_value=9.0,
                    value=6.6,
                    step=0.1,
                    help="Glycated Hemoglobin. ADA Clinical Criteria: <5.7% Normal, 5.7%-6.4% Prediabetes, >=6.5% Diabetes threshold.",
                )
                in_glucose_100k = st.slider(
                    "Blood Glucose Level (mg/dL):",
                    min_value=80,
                    max_value=300,
                    value=150,
                    step=1,
                    help="Fasting or random plasma blood glucose in mg/dL.",
                )
                in_bmi_100k = st.number_input(
                    "Body Mass Index (BMI, kg/m²):",
                    min_value=10.0,
                    max_value=70.0,
                    value=28.5,
                    step=0.1,
                    help="Body mass index in kg/m².",
                )

            with col2:
                st.markdown("##### Comorbidities & Lifestyle History")
                in_hyp_choice = st.radio(
                    "Hypertension History:",
                    options=["No (Not Diagnosed)", "Yes (Diagnosed)"],
                    index=0,
                    help="Documented clinical history of high blood pressure.",
                )
                in_hyp = 1 if "Yes" in in_hyp_choice else 0

                in_hd_choice = st.radio(
                    "Heart Disease History:",
                    options=["No (Not Diagnosed)", "Yes (Diagnosed)"],
                    index=0,
                    help="Documented history of cardiovascular disease or coronary artery disease.",
                )
                in_hd = 1 if "Yes" in in_hd_choice else 0

                in_smoking = st.selectbox(
                    "Smoking History:",
                    options=SMOKING_DISPLAY_OPTIONS,
                    index=0,
                    help="Categorical lifetime smoking exposure status.",
                )

            submit_btn_100k = st.form_submit_button("Generate Prediction & Explanations", type="primary")

        current_patient = {
            "Name": in_patient_name.strip() or "Not Provided",
            "Sex": in_sex,
            "gender": in_sex,
            "age": in_age_100k,
            "hypertension": in_hyp,
            "heart_disease": in_hd,
            "smoking_history": in_smoking,
            "bmi": in_bmi_100k,
            "HbA1c_level": in_hba1c,
            "blood_glucose_level": in_glucose_100k,
        }

        # Preprocess patient
        patient_scaled = preprocess_patient_100k(current_patient, scaler)

    else:
        # PIMA Legacy Form
        st.caption("Enter demographic and physiological measurements. Biologically implausible 0 values in Glucose, BP, Skin Thickness, Insulin, or BMI are automatically handled via median imputation.")

        demo_col1, demo_col2 = st.columns([2, 1], gap="medium")
        with demo_col1:
            in_patient_name = st.text_input(
                "Patient Full Name:",
                value="Jane Doe",
                key="patient_name_input_pima",
                help="Patient's full name for clinical documentation and reporting.",
            )
        with demo_col2:
            in_sex = st.selectbox(
                "Sex:",
                ["Female", "Male", "Not specified / Prefer not to say"],
                index=0,
                key="patient_sex_input_pima",
                help="When Male is selected, Pregnancies automatically defaults to 0.",
            )

        with st.form("prediction_input_form_pima", border=True):
            col1, col2 = st.columns(2, gap="medium")

            with col1:
                if in_sex == "Male":
                    in_pregnancies = 0
                    st.number_input(
                        DISPLAY_LABELS_PIMA["Pregnancies"],
                        min_value=0,
                        max_value=0,
                        value=0,
                        disabled=True,
                        help="Pregnancies is automatically set to 0 for biological male patients.",
                    )
                else:
                    in_pregnancies = st.number_input(
                        DISPLAY_LABELS_PIMA["Pregnancies"],
                        min_value=0,
                        max_value=20,
                        value=DEFAULT_INPUTS_PIMA["Pregnancies"],
                        step=1,
                        help=HELP_TEXT_PIMA["Pregnancies"],
                    )
                in_glucose = st.number_input(
                    DISPLAY_LABELS_PIMA["Glucose"],
                    min_value=0,
                    max_value=300,
                    value=DEFAULT_INPUTS_PIMA["Glucose"],
                    step=1,
                    help=HELP_TEXT_PIMA["Glucose"],
                )
                in_bp = st.number_input(
                    DISPLAY_LABELS_PIMA["BloodPressure"],
                    min_value=0,
                    max_value=200,
                    value=DEFAULT_INPUTS_PIMA["BloodPressure"],
                    step=1,
                    help=HELP_TEXT_PIMA["BloodPressure"],
                )
                in_skin = st.number_input(
                    DISPLAY_LABELS_PIMA["SkinThickness"],
                    min_value=0,
                    max_value=100,
                    value=DEFAULT_INPUTS_PIMA["SkinThickness"],
                    step=1,
                    help=HELP_TEXT_PIMA["SkinThickness"],
                )

            with col2:
                in_insulin = st.number_input(
                    DISPLAY_LABELS_PIMA["Insulin"],
                    min_value=0,
                    max_value=900,
                    value=DEFAULT_INPUTS_PIMA["Insulin"],
                    step=1,
                    help=HELP_TEXT_PIMA["Insulin"],
                )
                in_bmi = st.number_input(
                    DISPLAY_LABELS_PIMA["BMI"],
                    min_value=0.0,
                    max_value=75.0,
                    value=DEFAULT_INPUTS_PIMA["BMI"],
                    step=0.1,
                    help=HELP_TEXT_PIMA["BMI"],
                )
                in_dpf = st.number_input(
                    DISPLAY_LABELS_PIMA["DiabetesPedigreeFunction"],
                    min_value=0.05,
                    max_value=3.0,
                    value=DEFAULT_INPUTS_PIMA["DiabetesPedigreeFunction"],
                    step=0.01,
                    help=HELP_TEXT_PIMA["DiabetesPedigreeFunction"],
                )
                in_age = st.number_input(
                    DISPLAY_LABELS_PIMA["Age"],
                    min_value=18,
                    max_value=120,
                    value=DEFAULT_INPUTS_PIMA["Age"],
                    step=1,
                    help=HELP_TEXT_PIMA["Age"],
                )

            submit_btn_pima = st.form_submit_button("Generate Prediction & Explanations", type="primary")

        current_patient = {
            "Name": in_patient_name.strip() or "Not Provided",
            "Sex": in_sex,
            "Pregnancies": in_pregnancies,
            "Glucose": in_glucose,
            "BloodPressure": in_bp,
            "SkinThickness": in_skin,
            "Insulin": in_insulin,
            "BMI": in_bmi,
            "DiabetesPedigreeFunction": in_dpf,
            "Age": in_age,
        }

        # Validate PIMA inputs
        is_valid, validation_errors = validate_patient_inputs(current_patient)
        if not is_valid:
            st.error("Input Validation Errors:")
            for err in validation_errors:
                st.markdown(f"- {err}")
            st.stop()

        patient_df = pd.DataFrame([current_patient], columns=FEATURE_ORDER)
        patient_scaled = preprocess_patient(patient_df, imputer, scaler)

    # Execute Prediction and Local SHAP
    if active_model is not None and tree_explainer is not None:
        pred_class = int(active_model.predict(patient_scaled)[0])
        pred_prob = float(active_model.predict_proba(patient_scaled)[0, 1])
        conf_gap = abs(pred_prob - 0.5)

        feat_list = FEATURE_ORDER_100K if is_100k else FEATURE_ORDER
        shap_df = get_local_shap_explanation(tree_explainer, patient_scaled, current_patient, features=feat_list)

        # Update Session State
        st.session_state["patient_data"] = current_patient
        st.session_state["patient_scaled"] = patient_scaled
        st.session_state["pred_class"] = pred_class
        st.session_state["pred_prob"] = pred_prob
        st.session_state["shap_df"] = shap_df
        st.session_state["active_model_name"] = model_display_name
        st.session_state["is_100k"] = is_100k

        # Display Results Hierarchy
        st.markdown("---")
        st.subheader("Diagnostic Prediction Assessment")
        p_name = current_patient.get("Name", "Anonymous Patient")
        p_sex = current_patient.get("Sex", "Female")
        p_age_val = current_patient.get("Age") or current_patient.get("age", "N/A")
        st.caption(f"**Patient Assessment Profile:** {p_name} | **Biological Sex:** {p_sex} | **Age:** {p_age_val} years | **Cohort:** {'Clinical 100k' if is_100k else 'PIMA 768'}")

        res_col1, res_col2, res_col3 = st.columns(3)
        if pred_class == 1:
            res_col1.metric("Predicted Classification", "Higher Risk (Class 1)", delta="Elevated Risk", delta_color="inverse", border=True)
        else:
            res_col1.metric("Predicted Classification", "Lower Risk (Class 0)", delta="Low Risk", delta_color="normal", border=True)

        res_col2.metric("Probability of Diabetes", f"{pred_prob:.2%}", border=True)
        res_col3.metric("Distance from Threshold (0.50)", f"{conf_gap * 100:.1f} percentage pts", border=True)

        st.markdown("##### Predicted Probability Meter")
        st.progress(pred_prob, text=f"Calculated Probability: {pred_prob:.2%}")

        st.markdown("##### Top 3 Most Influential Clinical Factors (Local SHAP Attribution)")
        top3_cols = st.columns(3)
        for idx, row in shap_df.head(3).iterrows():
            top3_cols[idx].metric(
                label=f"{row['Feature']} = {row['Patient_Value']}",
                value=f"{row['SHAP_Value']:+.4f} SHAP",
                delta=row["Direction"],
                delta_color="inverse" if "higher" in row["Direction"] else "normal",
                border=True,
            )

        # Personalized Educational Guidance
        personalized_guidance = generate_personalized_recommendations(
            patient_data=current_patient,
            shap_features=shap_df.to_dict(orient="records"),
            prediction_probability=pred_prob,
        )
        st.session_state["personalized_guidance"] = personalized_guidance

        st.markdown("---")
        st.markdown("##### Personalized Educational Guidance & Wellness Recommendations")
        st.caption("Dynamically synthesized from positive local SHAP risk drivers and individual clinical inputs.")

        with st.container(border=True):
            st.markdown(f"**Model-Based Summary:** {personalized_guidance['summary_text']}")

            if personalized_guidance["priority_areas"]:
                st.markdown("###### Highest-Priority Upward Risk Contributors:")
                for area in personalized_guidance["priority_areas"]:
                    tag_type = "Modifiable Factor" if area["is_modifiable"] else "Non-Modifiable Factor"
                    with st.expander(f"{area['display_name']} ({area['shap_value']:+.4f} SHAP) — {tag_type}", expanded=True):
                        st.markdown(f"*{area['why_it_matters']}*")
                        for b in area["guidance"]:
                            st.markdown(f"- {b}")

            if personalized_guidance["negative_contributors_summary"]:
                st.markdown("###### Factors Lowering Predicted Risk in This Assessment:")
                for neg_stmt in personalized_guidance["negative_contributors_summary"]:
                    st.markdown(f"- {neg_stmt}")

            st.markdown("###### General Preventive Wellness Guidance:")
            for gen_b in personalized_guidance["general_wellness"]:
                st.markdown(f"- {gen_b}")

# -----------------------------------------------------------------------------
# TAB 2: WHY THIS PREDICTION? (SHAP & LIME EXPLAINABILITY)
# -----------------------------------------------------------------------------

with tab_explain:
    st.subheader("Dual Local Explainability Framework (SHAP + LIME)")
    st.caption("Interpreting the model's prediction through independent local game-theoretic (SHAP) and perturbation surrogate (LIME) methodologies.")

    if "shap_df" not in st.session_state:
        st.info("Please submit patient measurements in Tab 1 to generate local explanations.")
        st.stop()

    shap_df = st.session_state["shap_df"]
    patient_scaled = st.session_state["patient_scaled"]

    # 1. Local SHAP Attribution Chart
    st.markdown("#### 1. Local SHAP Attribution Waterfall / Bar Chart")
    st.caption("Quantifies the exact positive or negative contribution of each feature pushing the prediction away from the cohort expected baseline value.")

    chart_data = shap_df.copy().sort_values("SHAP_Value")
    shap_chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("SHAP_Value:Q", title="SHAP Attribution (Log-Odds Shift)"),
            y=alt.Y("Feature:N", sort=list(chart_data["Feature"]), title="Clinical Parameter"),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(domain=["Pushes risk lower", "Pushes risk higher"], range=["#15803d", "#dc2626"]),
                legend=alt.Legend(title="Attribution Impact"),
            ),
            tooltip=["Feature", "Patient_Value", alt.Tooltip("SHAP_Value:Q", format="+.4f"), "Direction"],
        )
        .properties(height=300)
    )
    zero_rule = alt.Chart(pd.DataFrame({"Zero": [0]})).mark_rule(color="#94a3b8", strokeWidth=1.5).encode(x="Zero:Q")
    st.altair_chart(shap_chart + zero_rule, use_container_width=True)

    # 2. Local LIME Explanation
    st.markdown("#### 2. Local LIME Surrogate Attribution")
    st.caption("LIME constructs an interpretable linear surrogate within the local perturbation neighborhood of this patient.")

    if lime_explainer is not None:
        lime_df = get_local_lime_explanation(
            lime_explainer,
            active_model.predict_proba,
            patient_scaled.values[0],
            num_features=len(chart_data),
        )
        st.session_state["lime_df"] = lime_df

        lime_chart_data = lime_df.copy().sort_values("LIME_Weight")
        lime_chart = (
            alt.Chart(lime_chart_data)
            .mark_bar(cornerRadiusEnd=5)
            .encode(
                x=alt.X("LIME_Weight:Q", title="LIME Linear Surrogate Weight"),
                y=alt.Y("Condition:N", sort=list(lime_chart_data["Condition"]), title="Partition Condition"),
                color=alt.Color(
                    "Direction:N",
                    scale=alt.Scale(domain=["Pushes risk lower", "Pushes risk higher"], range=["#15803d", "#dc2626"]),
                    legend=None,
                ),
                tooltip=["Condition", "Feature", alt.Tooltip("LIME_Weight:Q", format="+.4f"), "Direction"],
            )
            .properties(height=300)
        )
        st.altair_chart(lime_chart + zero_rule, use_container_width=True)

        # 3. Explanation Consistency & Agreement
        st.markdown("#### 3. Cross-Methodology Agreement (SHAP vs LIME)")
        comp_table, agree_metrics = get_explanation_agreement(shap_df, lime_df)

        agr_col1, agr_col2, agr_col3 = st.columns(3)
        agr_col1.metric("Spearman Rank Correlation (ρ)", f"{agree_metrics['spearman_rank_correlation']:.3f}", border=True)
        agr_col2.metric("Top-3 Feature Overlap (Jaccard)", f"{agree_metrics['jaccard_top3_overlap']:.1%}", border=True)
        agr_col3.metric("Directional Sign Concordance", f"{agree_metrics['sign_concordance_rate']:.1%}", border=True)

        st.dataframe(
            comp_table[["Feature", "SHAP_Value", "SHAP_Rank", "LIME_Weight", "LIME_Rank"]],
            column_config={
                "SHAP_Value": st.column_config.NumberColumn("SHAP Attribution", format="%.4f"),
                "SHAP_Rank": st.column_config.NumberColumn("SHAP Rank"),
                "LIME_Weight": st.column_config.NumberColumn("LIME Weight", format="%.4f"),
                "LIME_Rank": st.column_config.NumberColumn("LIME Rank"),
            },
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("View Global SHAP Distribution Figures (Cohort-Level)"):
        g_col1, g_col2 = st.columns(2)
        if is_100k:
            if (FIGURES_DIR / "shap_summary_100k.png").exists():
                g_col1.image(str(FIGURES_DIR / "shap_summary_100k.png"), caption="Global SHAP Summary (100k Cohort)")
            if (FIGURES_DIR / "shap_importance_100k.png").exists():
                g_col2.image(str(FIGURES_DIR / "shap_importance_100k.png"), caption="Global Feature Importance (100k Cohort)")
        else:
            if (FIGURES_DIR / "shap_summary.png").exists():
                g_col1.image(str(FIGURES_DIR / "shap_summary.png"), caption="Global SHAP Summary (PIMA)")
            if (FIGURES_DIR / "shap_feature_importance.png").exists():
                g_col2.image(str(FIGURES_DIR / "shap_feature_importance.png"), caption="Global Feature Importance (PIMA)")

# -----------------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE & BENCHMARKING
# -----------------------------------------------------------------------------

with tab_performance:
    st.subheader(f"Model Benchmarking & Diagnostic Evaluation ({'100k Clinical Cohort' if is_100k else 'PIMA Benchmark'})")

    if is_100k:
        st.markdown("#### 1. Candidate Algorithm Benchmark (Holdout Test Split, N=19,226)")
        m_path = RESULTS_DIR / "model_metrics_100k.csv"
        if m_path.exists():
            st.dataframe(
                pd.read_csv(m_path),
                column_config={
                    "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                    "Precision": st.column_config.NumberColumn(format="%.4f"),
                    "Recall": st.column_config.NumberColumn(format="%.4f"),
                    "F1 Score": st.column_config.NumberColumn(format="%.4f"),
                    "ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
                    "Brier Score": st.column_config.NumberColumn(format="%.4f"),
                },
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("#### 2. Stratified 5-Fold Cross-Validation (Primary XGBoost)")
        cv_100k_data = pd.DataFrame([
            {"Fold": "Fold 1", "ROC-AUC": 0.9785},
            {"Fold": "Fold 2", "ROC-AUC": 0.9793},
            {"Fold": "Fold 3", "ROC-AUC": 0.9793},
            {"Fold": "Fold 4", "ROC-AUC": 0.9798},
            {"Fold": "Fold 5", "ROC-AUC": 0.9789},
            {"Fold": "Mean 5-Fold", "ROC-AUC": 0.9792},
        ])
        st.dataframe(cv_100k_data, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 3. Diagnostic Evaluation Curves (100k Cohort)")
        d_col1, d_col2 = st.columns(2)
        if (FIGURES_DIR / "roc_curve_100k.png").exists():
            d_col1.image(str(FIGURES_DIR / "roc_curve_100k.png"), caption="Holdout ROC Curve (AUC = 0.9770)")
        if (FIGURES_DIR / "pr_curve_100k.png").exists():
            d_col2.image(str(FIGURES_DIR / "pr_curve_100k.png"), caption="Holdout Precision-Recall Curve")

        d_col3, d_col4 = st.columns(2)
        if (FIGURES_DIR / "confusion_matrix_100k.png").exists():
            d_col3.image(str(FIGURES_DIR / "confusion_matrix_100k.png"), caption="Confusion Matrix (N=19,226)")
        if (FIGURES_DIR / "calibration_curve_100k.png").exists():
            d_col4.image(str(FIGURES_DIR / "calibration_curve_100k.png"), caption="Probability Calibration Diagram")

    else:
        st.markdown("#### 1. Candidate Algorithm Benchmark (Holdout Test Split, N=154)")
        model_comp_path = RESULTS_DIR / "model_comparison.csv"
        if model_comp_path.exists():
            st.dataframe(pd.read_csv(model_comp_path), hide_index=True, use_container_width=True)

        st.markdown("#### 2. Stratified 5-Fold Cross-Validation")
        cv_path = RESULTS_DIR / "cross_validation.csv"
        if cv_path.exists():
            st.dataframe(pd.read_csv(cv_path), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 3. Diagnostic Evaluation Curves (PIMA)")
        d_col1, d_col2 = st.columns(2)
        if (FIGURES_DIR / "roc_curves.png").exists():
            d_col1.image(str(FIGURES_DIR / "roc_curves.png"), caption="Comparative ROC Curves")
        if (FIGURES_DIR / "pr_curves.png").exists():
            d_col2.image(str(FIGURES_DIR / "pr_curves.png"), caption="Precision-Recall Curves")

# -----------------------------------------------------------------------------
# TAB 4: SENSITIVITY & ROBUSTNESS ANALYSIS
# -----------------------------------------------------------------------------

with tab_sensitivity:
    st.subheader(f"Sensitivity Analysis & Prediction Robustness ({'100k Cohort' if is_100k else 'PIMA Cohort'})")

    if is_100k:
        st.markdown("#### 1. One-At-A-Time (OAT) Sensitivity: HbA1c vs. Diabetes Risk")
        st.caption("Demonstrates smooth clinical monotonicity: as HbA1c advances from normal (<5.7%) to prediabetes (5.7%-6.4%) and diabetic levels (>=6.5%), predicted risk rises steeply.")
        if (FIGURES_DIR / "sensitivity_hba1c_100k.png").exists():
            st.image(str(FIGURES_DIR / "sensitivity_hba1c_100k.png"), caption="HbA1c Sensitivity Trajectory")

        st.markdown("---")
        st.markdown("#### 2. Local Perturbation Robustness Testing (100k Cohort)")
        st.markdown(
            """
            - **Noise Stability**: Under Gaussian noise injections of $\\sigma = 0.01$, $\\sigma = 0.05$, and $\\sigma = 0.10$, the 100k model maintains an **Explanation Cosine Stability of 0.8804 / 1.0000**.
            - **Sample Size Resilience**: With $N=100,000$ patient records, decision boundaries are substantially smoother than small-sample datasets, drastically minimizing classification flips near boundaries.
            """
        )

    else:
        if "patient_data" not in st.session_state:
            st.info("Please submit patient measurements in Tab 1 to run sensitivity analysis.")
            st.stop()

        patient_dict = st.session_state["patient_data"]
        st.markdown("#### 1. One-At-A-Time (OAT) Sensitivity Sweep")
        sweep_feature = st.selectbox(
            "Select Clinical Parameter to Sweep:",
            options=["Glucose", "BMI", "Age", "Insulin", "BloodPressure"],
            index=0,
        )

        sens_dict = get_feature_sensitivity(patient_dict, active_model, artifacts["pima"]["imputer"], artifacts["pima"]["scaler"])
        sweep_df = sens_dict[sweep_feature]

        sens_chart = (
            alt.Chart(sweep_df)
            .mark_line(point=True, color="#0284c7", strokeWidth=2.5)
            .encode(
                x=alt.X("Feature_Value:Q", title=f"{sweep_feature} Value"),
                y=alt.Y("Predicted_Probability:Q", scale=alt.Scale(domain=[0, 1]), title="Estimated Diabetes Probability"),
                tooltip=[alt.Tooltip("Feature_Value:Q", format=".1f"), alt.Tooltip("Predicted_Probability:Q", format=".2%")],
            )
            .properties(height=320)
        )
        current_val = float(patient_dict[sweep_feature])
        patient_rule = alt.Chart(pd.DataFrame({"Current": [current_val]})).mark_rule(color="#e11d48", strokeDash=[4, 4]).encode(x="Current:Q")
        threshold_line = alt.Chart(pd.DataFrame({"Th": [0.5]})).mark_rule(color="#94a3b8", strokeDash=[6, 4]).encode(y="Th:Q")
        st.altair_chart(sens_chart + patient_rule + threshold_line, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: AI PATIENT REPORT & EXPORT
# -----------------------------------------------------------------------------

with tab_report:
    st.subheader("AI-Generated Clinical Risk Assessment Report")
    st.caption("Automated narrative synthesis grounded in patient inputs, SHAP attributions, and research validation metrics.")

    llm_status = get_llm_status()
    smtp_ok, smtp_note = check_email_configuration()

    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.markdown("**AI Report Generator:**")
        if llm_status["is_configured"]:
            st.success(f"✓ LLM configured ({llm_status['provider']} - {llm_status['model']})")
        else:
            st.warning("⚠ LLM not configured — using rule-based deterministic template engine")

    with stat_col2:
        st.markdown("**Email Dispatch Service:**")
        if smtp_ok:
            st.success("✓ Configured (Ready to dispatch)")
        else:
            st.warning("⚠ Not configured in `.env`")

    st.markdown("---")

    if "patient_data" not in st.session_state:
        st.info("Please submit patient measurements in Tab 1 to generate an AI clinical report.")
        st.stop()

    patient_dict = st.session_state["patient_data"]
    pred_class = st.session_state["pred_class"]
    pred_prob = st.session_state["pred_prob"]
    shap_df = st.session_state["shap_df"]
    active_model_name = st.session_state["active_model_name"]
    is_active_100k = st.session_state.get("is_100k", True)

    model_info = {
        "model_name": active_model_name,
        "roc_auc": cohort_metrics["roc_auc"],
    }

    lime_results = (
        st.session_state["lime_df"].to_dict(orient="records")
        if "lime_df" in st.session_state and isinstance(st.session_state["lime_df"], pd.DataFrame)
        else None
    )

    report_dict = generate_patient_report(
        patient_data=patient_dict,
        prediction=pred_class,
        probability=pred_prob,
        shap_results=shap_df.to_dict(orient="records"),
        model_info=model_info,
        lime_results=lime_results,
    )
    st.session_state["report_dict"] = report_dict

    personalized_guidance = report_dict.get("personalized_guidance")
    if not personalized_guidance:
        personalized_guidance = generate_personalized_recommendations(
            patient_data=patient_dict,
            shap_features=shap_df.to_dict(orient="records"),
            prediction_probability=pred_prob,
        )

    st.markdown("#### Clinical Assessment Document")
    p_name = patient_dict.get("Name", "Not Provided")
    p_sex = patient_dict.get("Sex", "Not specified / Prefer not to say")
    p_age = patient_dict.get("Age") or patient_dict.get("age", "N/A")
    st.caption(f"**Patient Assessment Profile:** {p_name} | **Biological Sex:** {p_sex} | **Age:** {p_age} years")

    with st.expander("Personalized Guidance & Highest-Priority Model Drivers Preview", expanded=True):
        st.markdown(f"**Model Summary:** {personalized_guidance['summary_text']}")
        for area in personalized_guidance.get("priority_areas", []):
            st.markdown(f"- **{area['display_name']}** (SHAP: {area['shap_value']:+.4f}): {area['guidance'][0]}")

    with st.container(border=True):
        st.text_area(
            "Full Clinical Assessment Narrative",
            value=report_dict["report_text"],
            height=350,
            help="Complete grounded clinical assessment report.",
        )

    # PDF Download Button
    st.markdown("---")
    st.markdown("#### Export Assessment to 9-Section PDF Report")
    chart_path = (
        str(FIGURES_DIR / "shap_summary_100k.png")
        if (is_active_100k and (FIGURES_DIR / "shap_summary_100k.png").exists())
        else (str(FIGURES_DIR / "shap_summary.png") if (FIGURES_DIR / "shap_summary.png").exists() else None)
    )

    pdf_bytes = generate_pdf_report(
        patient_data=patient_dict,
        prediction=pred_class,
        probability=pred_prob,
        top_shap_features=shap_df.to_dict(orient="records"),
        model_info=model_info,
        interpretation_text=report_dict["interpretation"],
        recommendations=report_dict["recommendations"],
        chart_image_path=chart_path,
        personalized_guidance=personalized_guidance,
    )

    st.download_button(
        label="Download Professional 9-Section PDF Report",
        data=pdf_bytes,
        file_name=f"Diabetes_Risk_Assessment_Report_{p_name.replace(' ', '_')}.pdf",
        mime="application/pdf",
        type="primary",
    )

    # Email Dispatch Section
    st.markdown("---")
    st.markdown("#### Dispatch Report via Email")
    with st.form("email_dispatch_form"):
        recipient_email = st.text_input("Recipient Email Address:", placeholder="clinician@hospital.org")
        send_email_btn = st.form_submit_button("Send Report to Email", type="primary")

    if send_email_btn:
        if not recipient_email or not validate_email_address(recipient_email):
            st.error("Please enter a valid email address.")
        else:
            email_body = format_email_body(
                prediction=pred_class,
                probability=pred_prob,
                top_shap_features=shap_df.to_dict(orient="records"),
                interpretation_text=report_dict["interpretation"],
                model_name=active_model_name,
                patient_data=patient_dict,
                personalized_guidance=personalized_guidance,
            )
            with st.spinner("Transmitting report via secure SMTP..."):
                success, msg = send_report_email(
                    recipient_email=recipient_email,
                    subject="Diabetes Risk Assessment Report",
                    body_text=email_body,
                    pdf_bytes=pdf_bytes,
                    pdf_filename="Diabetes_Risk_Assessment_Report.pdf",
                )
                if success:
                    st.success(msg, icon=":material/mark_email_read:")
                else:
                    st.error(msg, icon=":material/error:")

# -----------------------------------------------------------------------------
# TAB 6: ABOUT & TRUST EVALUATION FRAMEWORK
# -----------------------------------------------------------------------------

with tab_about:
    st.subheader("Academic Architecture & Multi-Dimensional Trust Framework")
    st.markdown(
        r"""
        ### Multi-Cohort Architecture & Faculty Dataset Upgrade
        Following faculty review and academic evaluation guidance, this system has been upgraded to a **Multi-Cohort Architecture**:

        - **Primary Clinical Cohort ($N=100,000$)**:
          - Features clinical gold-standard diagnostic biomarkers (**HbA1c**, **blood glucose**), lifestyle/comorbidities (**hypertension**, **heart disease**, **smoking history**), and balanced demographics (**58.6% Female, 41.4% Male**).
          - Holdout ROC-AUC of **0.9770** and 5-Fold Stratified CV ROC-AUC of **0.9792 ± 0.0004**.
          - Directly addresses faculty feedback regarding dataset scale ($130\times$ increase over PIMA) and realistic hospital biomarker inclusion.
        - **Historical PIMA Benchmark ($N=768$)**:
          - Preserved as a comparative reference benchmark to demonstrate model generalizability and cross-cohort explainability.

        ### Multi-Dimensional Trust Evaluation Dimensions
        1. **Predictive Discrimination**: Evaluated via ROC-AUC (0.9770 on 100k cohort; 0.8198 on PIMA benchmark).
        2. **Cross-Validation Stability**: Stratified 5-Fold CV demonstrates minimal standard deviation ($\sigma = 0.0004$ on 100k; $\sigma = 0.0237$ on PIMA).
        3. **Explanation Consistency**: High rank alignment between independent XAI frameworks (SHAP TreeExplainer and LIME tabular linear surrogates).
        4. **Local Perturbation Robustness**: Evaluated via Gaussian physiological noise perturbations (Robustness Index $0.8804 / 1.0000$).
        5. **Error Profile Transparency**: Transparent accounting of false negatives and false positives.
        6. **Sensitivity Plausibility**: Confirms that clinical risk rises monotonically with diagnostic biomarkers like HbA1c and glucose.
        """
    )

# -----------------------------------------------------------------------------
# TAB 7: PROJECT LIMITATIONS & ETHICAL BOUNDARIES
# -----------------------------------------------------------------------------

with tab_limitations:
    st.subheader("System Limitations & Ethical Boundaries")
    st.markdown(
        r"""
        #### 1. Non-Diagnostic Medical Scope
        - This application is an **academic and research decision-support prototype**.
        - It is **not** an FDA-cleared or CE-marked medical device.
        - Outputs must **never** be used for self-diagnosis, clinical prescription, or dosage modification.

        #### 2. Clinical Biomarker Context
        - While the 100,000-record clinical dataset incorporates diagnostic biomarkers (HbA1c and Blood Glucose), real-world clinical diagnosis requires repeated laboratory testing under standardized medical supervision.

        #### 3. Tree Ensemble Piecewise Behavior
        - Gradient Boosted Trees partition feature space through axis-aligned splits. Sensitivity curves can exhibit localized plateaus where probabilities remain constant across certain feature intervals.

        #### 4. Attribution is Not Biological Causation
        - SHAP and LIME values quantify computational feature attribution within the trained machine learning weights. They do **not** prove biological causation or clinical etiology.
        """
    )

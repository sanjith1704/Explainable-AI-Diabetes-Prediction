from pathlib import Path

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


FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

DISPLAY_LABELS = {
    "Pregnancies": "Pregnancies (count)",
    "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Blood pressure (mmHg)",
    "SkinThickness": "Skin thickness (mm)",
    "Insulin": "Insulin (muU/mL)",
    "BMI": "BMI (kg/m^2)",
    "DiabetesPedigreeFunction": "Diabetes pedigree function (unitless)",
    "Age": "Age (years)",
}

HELP_TEXT = {
    "Pregnancies": "Number of prior pregnancies.",
    "Glucose": "Plasma glucose concentration entered in mg/dL.",
    "BloodPressure": "Diastolic blood pressure in mmHg.",
    "SkinThickness": "Triceps skin fold thickness in millimeters.",
    "Insulin": "Two-hour serum insulin in muU/mL. If unknown, 0 is treated as missing and imputed.",
    "BMI": "Body mass index in kg/m^2. If unknown, 0 is treated as missing and imputed.",
    "DiabetesPedigreeFunction": "Family-history-related score used by the dataset.",
    "Age": "Patient age in completed years.",
}

INVALID_ZERO_FEATURES = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

DEFAULT_INPUT = {
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


st.set_page_config(
    page_title="Explainable AI diabetes prediction",
    page_icon=":material/health_and_safety:",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgboost.pkl")
    scaler = joblib.load("models/scaler.pkl")
    imputer = joblib.load("models/imputer.pkl")
    explainer = shap.TreeExplainer(model)
    return model, scaler, imputer, explainer


@st.cache_data
def load_validation_snapshot():
    model = joblib.load("models/xgboost.pkl")
    x_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze("columns")

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "samples": int(len(y_test)),
    }


def build_input_frame(values):
    return pd.DataFrame([[values[name] for name in FEATURE_ORDER]], columns=FEATURE_ORDER)


def preprocess_input(input_frame, imputer, scaler):
    cleaned = input_frame.copy()
    cleaned[INVALID_ZERO_FEATURES] = cleaned[INVALID_ZERO_FEATURES].replace(0, np.nan)

    imputed = pd.DataFrame(
        imputer.transform(cleaned),
        columns=FEATURE_ORDER,
        index=cleaned.index,
    )

    scaled = pd.DataFrame(
        scaler.transform(imputed.values),
        columns=FEATURE_ORDER,
        index=cleaned.index,
    )
    return scaled


def extract_shap_values(explanation):
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.ndim == 2:
        values = values[0]
    return values.astype(float)


def format_feature_value(feature_name, value):
    if feature_name in {"Pregnancies", "Age"}:
        return f"{int(value)}"
    if feature_name in {"Glucose", "BloodPressure", "SkinThickness", "Insulin"}:
        return f"{float(value):.0f}"
    if feature_name == "BMI":
        return f"{float(value):.1f}"
    if feature_name == "DiabetesPedigreeFunction":
        return f"{float(value):.2f}"
    return f"{float(value):.2f}"


def build_contribution_frame(raw_input, shap_values):
    contribution_df = pd.DataFrame(
        {
            "Feature": FEATURE_ORDER,
            "Display": [DISPLAY_LABELS[name] for name in FEATURE_ORDER],
            "Value": raw_input.iloc[0].tolist(),
            "SHAP Value": shap_values,
        }
    )
    contribution_df["Absolute Importance"] = contribution_df["SHAP Value"].abs()
    contribution_df["Direction"] = np.where(
        contribution_df["SHAP Value"] >= 0,
        "Pushes risk higher",
        "Pushes risk lower",
    )
    contribution_df["Signed Contribution"] = contribution_df["SHAP Value"].round(4)
    contribution_df["Formatted Value"] = [
        format_feature_value(feature, value)
        for feature, value in zip(contribution_df["Feature"], contribution_df["Value"], strict=False)
    ]
    return contribution_df.sort_values("Absolute Importance", ascending=False)


def render_probability_chart(probability):
    chart_data = pd.DataFrame(
        {
            "Label": ["Current patient"],
            "Probability": [probability],
            "Band": ["Higher risk" if probability >= 0.5 else "Lower risk"],
        }
    )
    threshold_data = pd.DataFrame({"Threshold": [0.5]})

    bar_chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadius=10, size=42)
        .encode(
            x=alt.X(
                "Probability:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
                title="Predicted probability of the diabetes class",
            ),
            y=alt.Y("Label:N", axis=None),
            color=alt.Color(
                "Band:N",
                scale=alt.Scale(
                    domain=["Lower risk", "Higher risk"],
                    range=["#0f766e", "#b42318"],
                ),
                legend=None,
            ),
            tooltip=[alt.Tooltip("Probability:Q", format=".2%")],
        )
    )

    threshold_rule = (
        alt.Chart(threshold_data)
        .mark_rule(color="#667085", strokeDash=[6, 4], strokeWidth=2)
        .encode(x="Threshold:Q")
    )

    label_layer = (
        alt.Chart(chart_data)
        .mark_text(align="left", baseline="middle", dx=10, color="#101828")
        .encode(
            x="Probability:Q",
            y=alt.Y("Label:N", axis=None),
            text=alt.Text("Probability:Q", format=".2%"),
        )
    )

    st.altair_chart(bar_chart + threshold_rule + label_layer, width="stretch")


def render_shap_chart(contribution_df):
    chart_df = contribution_df.sort_values("SHAP Value").copy()

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X(
                "SHAP Value:Q",
                title="SHAP value",
                axis=alt.Axis(format=".3f"),
            ),
            y=alt.Y("Display:N", title=None, sort=list(chart_df["Display"])),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(
                    domain=["Pushes risk lower", "Pushes risk higher"],
                    range=["#0f766e", "#b42318"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Display:N", title="Feature"),
                alt.Tooltip("Formatted Value:N", title="Input value"),
                alt.Tooltip("SHAP Value:Q", format=".4f"),
            ],
        )
    )

    zero_rule = alt.Chart(pd.DataFrame({"Zero": [0]})).mark_rule(color="#98a2b3").encode(x="Zero:Q")
    st.altair_chart(chart + zero_rule, width="stretch")


def render_global_explainability():
    image_specs = [
        (
            FIGURES_DIR / "shap_feature_importance.png",
            "Global SHAP feature importance from the notebook analysis",
        ),
        (
            FIGURES_DIR / "shap_summary.png",
            "Global SHAP summary plot from the notebook analysis",
        ),
    ]

    shown_any = False
    for image_path, caption in image_specs:
        if image_path.exists():
            shown_any = True
            st.image(str(image_path), caption=caption, width="stretch")

    if not shown_any:
        st.caption("Precomputed global SHAP figures were not found in `results/figures`.")


model, scaler, imputer, explainer = load_artifacts()
validation = load_validation_snapshot()


with st.sidebar:
    st.title("Project summary")
    st.caption("Educational and research-focused diabetes risk prediction with local SHAP explanations.")

    st.markdown(
        """
        **Deployed pipeline**

        1. Patient input
        2. Missing-style zero handling
        3. Median imputation
        4. Standard scaling
        5. XGBoost probability
        6. SHAP explanation
        """
    )

    st.markdown(
        """
        **Holdout validation**

        - Accuracy: 74.03%
        - Precision: 65.22%
        - Recall: 55.56%
        - F1 score: 60.00%
        - ROC-AUC: 81.98%
        """
    )

    st.warning(
        "This application is not a medical diagnostic system and must not replace professional medical advice.",
        icon=":material/warning:",
    )


st.title("Explainable AI diabetes prediction")
st.caption(
    "The app predicts the probability of the diabetes class with the saved XGBoost model and explains the result with SHAP. SHAP values describe model behavior for this input; they do not prove medical causation."
)

overview_cols = st.columns(3)
overview_cols[0].metric("Model", "XGBoost", border=True)
overview_cols[1].metric("Input features", "8", border=True)
overview_cols[2].metric("Test samples", str(validation["samples"]), border=True)

with st.container(border=True):
    st.subheader("Enter patient features")

    with st.form("patient_inputs", border=False):
        left_col, right_col = st.columns(2, gap="large")

        with left_col:
            pregnancies = st.number_input(
                DISPLAY_LABELS["Pregnancies"],
                min_value=0,
                max_value=20,
                value=DEFAULT_INPUT["Pregnancies"],
                step=1,
                help=HELP_TEXT["Pregnancies"],
            )
            glucose = st.number_input(
                DISPLAY_LABELS["Glucose"],
                min_value=0,
                max_value=250,
                value=DEFAULT_INPUT["Glucose"],
                step=1,
                help=HELP_TEXT["Glucose"],
            )
            blood_pressure = st.number_input(
                DISPLAY_LABELS["BloodPressure"],
                min_value=0,
                max_value=150,
                value=DEFAULT_INPUT["BloodPressure"],
                step=1,
                help=HELP_TEXT["BloodPressure"],
            )
            skin_thickness = st.number_input(
                DISPLAY_LABELS["SkinThickness"],
                min_value=0,
                max_value=100,
                value=DEFAULT_INPUT["SkinThickness"],
                step=1,
                help=HELP_TEXT["SkinThickness"],
            )

        with right_col:
            insulin = st.number_input(
                DISPLAY_LABELS["Insulin"],
                min_value=0,
                max_value=900,
                value=DEFAULT_INPUT["Insulin"],
                step=1,
                help=HELP_TEXT["Insulin"],
            )
            bmi = st.number_input(
                DISPLAY_LABELS["BMI"],
                min_value=0.0,
                max_value=70.0,
                value=DEFAULT_INPUT["BMI"],
                step=0.1,
                help=HELP_TEXT["BMI"],
            )
            diabetes_pedigree = st.number_input(
                DISPLAY_LABELS["DiabetesPedigreeFunction"],
                min_value=0.0,
                max_value=3.0,
                value=DEFAULT_INPUT["DiabetesPedigreeFunction"],
                step=0.01,
                help=HELP_TEXT["DiabetesPedigreeFunction"],
            )
            age = st.number_input(
                DISPLAY_LABELS["Age"],
                min_value=1,
                max_value=120,
                value=DEFAULT_INPUT["Age"],
                step=1,
                help=HELP_TEXT["Age"],
            )

        submitted = st.form_submit_button("Generate prediction")

    st.caption(
        "For Glucose, Blood pressure, Skin thickness, Insulin, and BMI, an entered value of 0 is treated as missing in the same way as the training pipeline before imputation and scaling."
    )


if submitted:
    input_values = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age,
    }

    raw_input = build_input_frame(input_values)
    processed_input = preprocess_input(raw_input, imputer, scaler)

    prediction = int(model.predict(processed_input)[0])
    probability = float(model.predict_proba(processed_input)[0, 1])
    confidence_gap = abs(probability - 0.5)

    shap_explanation = explainer(processed_input)
    shap_values = extract_shap_values(shap_explanation)
    contribution_df = build_contribution_frame(raw_input, shap_values)

    result_cols = st.columns(3)
    result_cols[0].metric(
        "Predicted class",
        "Higher risk" if prediction == 1 else "Lower risk",
        border=True,
    )
    result_cols[1].metric(
        "Diabetes probability",
        f"{probability * 100:.2f}%",
        border=True,
    )
    result_cols[2].metric(
        "Distance from 50% threshold",
        f"{confidence_gap * 100:.2f} points",
        border=True,
    )

    with st.container(border=True):
        st.subheader("Probability view")
        render_probability_chart(probability)
        st.progress(probability, text=f"Predicted probability: {probability * 100:.2f}%")

    explanation_tab, context_tab, limitations_tab = st.tabs(
        ["Patient explanation", "Model context", "Limitations"]
    )

    with explanation_tab:
        st.subheader("Why did the model make this prediction?")
        st.caption(
            "Positive SHAP values move the model toward the diabetes class. Negative SHAP values move it toward the non-diabetes class. Larger absolute values mean a stronger effect on this prediction."
        )

        render_shap_chart(contribution_df)

        st.dataframe(
            contribution_df[["Display", "Formatted Value", "Signed Contribution", "Direction"]],
            column_config={
                "Display": st.column_config.TextColumn("Feature"),
                "Formatted Value": st.column_config.TextColumn("Input value"),
                "Signed Contribution": st.column_config.NumberColumn("SHAP value", format="%.4f"),
                "Direction": st.column_config.TextColumn("Interpretation"),
            },
            hide_index=True,
            width="stretch",
        )

        st.subheader("Top contributing features")
        for _, row in contribution_df.head(5).iterrows():
            direction_text = "higher" if row["Direction"] == "Pushes risk higher" else "lower"
            st.markdown(
                f"- **{row['Display']}** = `{row['Formatted Value']}` pushed the model toward **{direction_text} risk** with SHAP value `{row['Signed Contribution']:.4f}`."
            )

        st.info(
            "Interpret these contributions as an explanation of the saved model's decision for this patient, not as proof that any factor medically caused diabetes.",
            icon=":material/info:",
        )

    with context_tab:
        st.subheader("Validation snapshot for the deployed model")
        validation_cols = st.columns(5)
        validation_cols[0].metric("Accuracy", f"{validation['accuracy'] * 100:.2f}%", border=True)
        validation_cols[1].metric("Precision", f"{validation['precision'] * 100:.2f}%", border=True)
        validation_cols[2].metric("Recall", f"{validation['recall'] * 100:.2f}%", border=True)
        validation_cols[3].metric("F1 score", f"{validation['f1'] * 100:.2f}%", border=True)
        validation_cols[4].metric("ROC-AUC", f"{validation['roc_auc'] * 100:.2f}%", border=True)

        st.caption(
            f"Confusion matrix on the saved holdout split: TN={validation['tn']}, FP={validation['fp']}, FN={validation['fn']}, TP={validation['tp']}."
        )

        with st.expander("Global explainability from the notebooks"):
            render_global_explainability()

        with st.expander("Why Age and Insulin can behave nonlinearly"):
            st.markdown(
                """
                - XGBoost is a tree ensemble, so its response is piecewise and interaction-driven rather than strictly monotonic.
                - A change in Age or Insulin can move a patient into different tree leaves, which may raise, lower, or flatten the predicted probability.
                - This should be described as model behavior learned from the dataset, not as a clinical law.
                """
            )

    with limitations_tab:
        st.subheader("Project limitations")
        st.markdown(
            """
            - The model is educational and research-oriented. It is not clinically validated.
            - Probability outputs reflect the saved model and dataset, not guaranteed real-world prevalence.
            - SHAP explains how the model used the input features. It does not establish medical causality.
            - Entering many unknown measurements as 0 increases reliance on imputation and should be interpreted cautiously.
            - The notebooks still contain duplicate experiments and stale exports that should be treated as audit targets rather than final evidence.
            """
        )

else:
    with st.container(border=True):
        st.subheader("What the app will show after prediction")
        st.markdown(
            """
            - The predicted probability of the diabetes class.
            - A higher-risk or lower-risk classification from the saved XGBoost model.
            - Local SHAP contributions ranked by magnitude.
            - A validation snapshot for the deployed artifact.
            """
        )

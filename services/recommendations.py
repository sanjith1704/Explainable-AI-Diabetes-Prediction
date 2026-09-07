"""
recommendations.py: Deterministic, SHAP-grounded educational guidance generator.

Generates personalized lifestyle and screening educational guidance based on:
1. Patient clinical input values.
2. Signed local SHAP feature attributions.
3. Ranking of positive SHAP contributors pushing model prediction toward higher risk.
4. Strict medical non-diagnostic guardrails (no medication advice, no crash diets,
   no claims of biological causation, non-modifiable factors clearly designated).
"""

from typing import Any, Dict, List, Optional


FEATURE_DISPLAY_NAMES = {
    "Glucose": "Fasting Blood Glucose",
    "BMI": "Body Mass Index (BMI)",
    "BloodPressure": "Diastolic Blood Pressure",
    "Insulin": "Serum Insulin",
    "Age": "Patient Age",
    "Pregnancies": "Pregnancy History",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function (Genetic History)",
    "SkinThickness": "Triceps Skinfold Thickness",
}

GENERAL_PREVENTIVE_WELLNESS = [
    "Adopt balanced dietary patterns emphasizing complex carbohydrates, vegetables, and whole foods.",
    "Ensure adequate dietary fiber intake to support healthy metabolic and digestive functioning.",
    "Prioritize lean protein sources and healthy unsaturated fats across daily meals.",
    "Limit excessive consumption of refined sugars, processed carbohydrates, and sugar-sweetened beverages.",
    "Maintain regular physical activity (targeting at least 150 minutes of moderate aerobic activity weekly as tolerated).",
    "Engage in routine preventive cardiovascular and metabolic health screenings with a healthcare provider.",
    "Consult qualified healthcare professionals before undertaking major dietary, physical, or lifestyle changes.",
]


def _get_guidance_for_feature(feature: str, val: Any, shap_val: float) -> Dict[str, Any]:
    """
    Returns educational guidance, model context, and modifiability status for a positive contributor.
    Strictly follows clinical safety and educational guidelines.
    """
    f_clean = feature.strip()

    if f_clean == "Glucose":
        return {
            "feature": "Glucose",
            "display_name": "Blood Glucose",
            "patient_value": val,
            "shap_value": shap_val,
            "is_modifiable": True,
            "why_it_matters": (
                "Glucose is one of the factors contributing most strongly to the model's estimated risk "
                f"for this assessment (positive SHAP contribution of {shap_val:+.4f})."
            ),
            "guidance": [
                "Discuss appropriate blood-glucose testing and routine monitoring with a qualified healthcare professional.",
                "If not recently evaluated, discuss fasting plasma glucose and/or glycated hemoglobin (HbA1c) screening with a physician.",
                "Encourage balanced meals emphasizing vegetables, high-fiber foods, whole grains, and appropriate lean protein sources.",
                "Limit excessive intake of refined carbohydrates, confectioneries, and sugar-sweetened beverages.",
                "Engage in regular physical activity appropriate for your baseline fitness and health status.",
                "Clinical Notice: Do not independently initiate or alter any medication, insulin therapy, or treatment regimen without physician oversight.",
            ],
        }

    elif f_clean == "BMI":
        return {
            "feature": "BMI",
            "display_name": "Body Mass Index (BMI)",
            "patient_value": val,
            "shap_value": shap_val,
            "is_modifiable": True,
            "why_it_matters": (
                "BMI contributed toward a higher model-estimated risk in this assessment "
                f"(positive SHAP contribution of {shap_val:+.4f})."
            ),
            "guidance": [
                "Encourage sustainable, nutritionally balanced eating patterns rather than restrictive or crash diets.",
                "Maintain regular physical activity appropriate to personal fitness level, joint health, and medical history.",
                "If weight management is clinically appropriate, discuss realistic and individualized goals with a healthcare professional or registered dietitian.",
                "Clinical Notice: The system does not promise that altering body weight will reduce the computational prediction by a specific amount.",
            ],
        }

    elif f_clean == "BloodPressure":
        return {
            "feature": "BloodPressure",
            "display_name": "Blood Pressure",
            "patient_value": val,
            "shap_value": shap_val,
            "is_modifiable": True,
            "why_it_matters": (
                "Blood pressure contributed toward a higher model-estimated risk in this assessment "
                f"(positive SHAP contribution of {shap_val:+.4f})."
            ),
            "guidance": [
                "Encourage routine blood-pressure monitoring under resting conditions.",
                "Discuss blood pressure readings with a healthcare professional if measurements are repeatedly elevated.",
                "Support cardiovascular health through a balanced dietary pattern (such as reduced sodium intake) and consistent physical activity.",
                "Clinical Notice: A single input measurement does not constitute a clinical diagnosis of hypertension.",
            ],
        }

    elif f_clean == "Insulin":
        return {
            "feature": "Insulin",
            "display_name": "Serum Insulin",
            "patient_value": val,
            "shap_value": shap_val,
            "is_modifiable": True,
            "why_it_matters": (
                "Insulin contributed toward a higher model-estimated risk in this assessment "
                f"(positive SHAP contribution of {shap_val:+.4f})."
            ),
            "guidance": [
                "Recommend discussing comprehensive metabolic and diabetes diagnostic testing with a healthcare professional.",
                "Explain that the computational model uses insulin strictly as a statistical predictive feature, not as clinical proof of insulin resistance or disease.",
                "Clinical Notice: Do not make any adjustments to prescribed insulin or diabetes medication without direct physician supervision.",
            ],
        }

    elif f_clean == "Age":
        return {
            "feature": "Age",
            "display_name": "Age",
            "patient_value": val,
            "shap_value": shap_val,
            "is_modifiable": False,
            "why_it_matters": (
                "Age is a non-modifiable factor that contributed toward the model's estimated risk "
                f"(positive SHAP contribution of {shap_val:+.4f})."
            ),
            "guidance": [
                "Maintain regular age-appropriate health screenings and cardiovascular wellness evaluations.",
                "Engage in consistent, low-impact physical activity suited for your age bracket and mobility.",
                "Maintain a balanced, nutrient-dense diet to support healthy metabolic function.",
                "Discuss regular clinical diabetes screening with a healthcare professional during routine check-ups.",
                "Clinical Notice: Age cannot be altered, and this guidance focuses purely on healthy aging and age-appropriate preventive vigilance.",
            ],
        }

    elif f_clean == "Pregnancies":
        return {
            "feature": "Pregnancies",
            "display_name": "Pregnancy History",
            "patient_value": val,
            "shap_value": shap_val,
            "is_modifiable": False,
            "why_it_matters": (
                "Pregnancy history is a historical/non-modifiable input in this assessment and cannot be changed. "
                f"Its contribution reflects how the trained model uses this feature (SHAP {shap_val:+.4f})."
            ),
            "guidance": [
                "Maintain awareness of personal obstetric and metabolic history, including any prior gestational diabetes screening.",
                "Participate in ongoing preventive wellness screenings with your healthcare provider.",
                "Clinical Notice: Historical reproductive count is non-modifiable; guidance is strictly focused on general metabolic wellness.",
            ],
        }

    elif f_clean == "DiabetesPedigreeFunction":
        return {
            "feature": "DiabetesPedigreeFunction",
            "display_name": "Diabetes Pedigree Function",
            "patient_value": val,
            "shap_value": shap_val,
            "is_modifiable": False,
            "why_it_matters": (
                "DiabetesPedigreeFunction contributed toward the model's estimated risk. "
                f"This feature represents family-history-related genetic information and is not directly modifiable (SHAP {shap_val:+.4f})."
            ),
            "guidance": [
                "Maintain active awareness of family medical history regarding diabetes and cardiovascular conditions.",
                "Share relevant family metabolic history during clinical discussions with your primary healthcare provider.",
                "Proactively maintain healthy lifestyle habits (nutrition and physical exercise) to optimize metabolic resilience.",
            ],
        }

    elif f_clean == "SkinThickness":
        return {
            "feature": "SkinThickness",
            "display_name": "Triceps Skinfold Thickness",
            "patient_value": val,
            "shap_value": shap_val,
            "is_modifiable": True,
            "why_it_matters": (
                "Skin thickness contributed toward the model's output in this assessment "
                f"(positive SHAP contribution of {shap_val:+.4f})."
            ),
            "guidance": [
                "Skinfold thickness is utilized by the algorithm as an anthropometric body composition proxy.",
                "No clinical diagnosis should be inferred from a standalone skinfold measurement.",
                "Focus on general wellness, regular physical activity, and well-rounded nutrition.",
            ],
        }

    # Fallback for unexpected feature name
    return {
        "feature": f_clean,
        "display_name": f_clean,
        "patient_value": val,
        "shap_value": shap_val,
        "is_modifiable": False,
        "why_it_matters": f"{f_clean} contributed toward a higher model-estimated risk (positive SHAP contribution of {shap_val:+.4f}).",
        "guidance": [
            f"Discuss the role of {f_clean} in your overall health assessment with a physician.",
            "Maintain general preventive lifestyle practices and regular wellness checkups.",
        ],
    }


def generate_personalized_recommendations(
    patient_data: Dict[str, Any],
    shap_features: List[Dict[str, Any]],
    prediction_probability: float = 0.5,
) -> Dict[str, Any]:
    """
    Core shared function to generate personalized educational recommendations.

    Used across:
    1. Streamlit Application UI
    2. PDF Report Generator
    3. Email Service Dispatch

    Parameters
    ----------
    patient_data : Dict[str, Any]
        Raw patient clinical inputs and metadata.
    shap_features : List[Dict[str, Any]]
        List of dicts with keys 'Feature', 'SHAP_Value', 'Patient_Value', 'Direction'.
    prediction_probability : float
        Estimated diabetes risk probability from the model.

    Returns
    -------
    Dict[str, Any] containing:
        - 'summary_text': Model-based overview of upward risk contributors.
        - 'priority_areas': Detailed structured priority areas for positive SHAP features.
        - 'negative_contributors_summary': Statements describing features that lowered predicted risk.
        - 'general_wellness': Standard preventive wellness guidelines.
        - 'all_personalized_bullets': Flat list of recommendation strings.
        - 'top_upward_features': List of positive SHAP feature names.
    """
    positive_contributors = []
    negative_contributors = []

    for item in shap_features:
        feat = item.get("Feature", "")
        shap_val = float(item.get("SHAP_Value", 0.0))
        val = item.get("Patient_Value", patient_data.get(feat, "N/A"))

        if shap_val > 0.0:
            positive_contributors.append((feat, val, shap_val))
        elif shap_val < 0.0:
            negative_contributors.append((feat, val, shap_val))

    # Sort positive contributors by absolute magnitude descending
    positive_contributors.sort(key=lambda x: abs(x[2]), reverse=True)
    # Sort negative contributors by absolute magnitude descending
    negative_contributors.sort(key=lambda x: abs(x[2]), reverse=True)

    # 2. Build Personalized Model-Based Summary
    top_upward_names = [f[0] for f in positive_contributors]
    if len(top_upward_names) >= 2:
        top_str = f"{top_upward_names[0]} and {top_upward_names[1]}"
    elif len(top_upward_names) == 1:
        top_str = f"{top_upward_names[0]}"
    else:
        top_str = "no individual features"

    if positive_contributors:
        summary_text = (
            f"Based on this assessment, the model's strongest upward contributors were {top_str}. "
            "These factors had positive local SHAP contributions and therefore pushed the model's prediction "
            "toward the diabetes-positive class. The following recommendations are educational guidance "
            "related to these model contributors."
        )
    else:
        summary_text = (
            "Based on this assessment, none of the clinical features exerted a positive local SHAP push "
            "toward the diabetes-positive class. All entered measurements either aligned with or lowered the "
            "baseline predicted risk. General preventive wellness recommendations are provided below."
        )

    # 3. Generate Highest-Priority Areas for upward contributors
    priority_areas = []
    all_personalized_bullets = []

    for feat, val, shap_val in positive_contributors:
        area_info = _get_guidance_for_feature(feat, val, shap_val)
        priority_areas.append(area_info)
        for bullet in area_info["guidance"]:
            all_personalized_bullets.append(f"[{area_info['display_name']}] {bullet}")

    # 4. Describe negative contributors without calling them "protective"
    negative_summary = []
    for feat, val, shap_val in negative_contributors:
        d_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
        negative_summary.append(
            f"{d_name} lowered the model's predicted risk for this particular assessment (SHAP {shap_val:+.4f})."
        )

    # 5. General preventive wellness guidance
    general_wellness = list(GENERAL_PREVENTIVE_WELLNESS)
    if prediction_probability >= 0.5:
        general_wellness.insert(
            0,
            "Given the elevated computational risk score, discuss confirmatory diagnostic laboratory testing (such as fasting glucose and HbA1c) with a licensed physician."
        )

    return {
        "summary_text": summary_text,
        "priority_areas": priority_areas,
        "negative_contributors_summary": negative_summary,
        "general_wellness": general_wellness,
        "all_personalized_bullets": all_personalized_bullets,
        "top_upward_features": top_upward_names,
    }

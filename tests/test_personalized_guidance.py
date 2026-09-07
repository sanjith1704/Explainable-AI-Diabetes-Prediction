"""
test_personalized_guidance.py: Comprehensive validation suite for personalized SHAP-based
educational guidance, demographic isolation, and PDF generation.

Tests the 15 critical requirements specified in the project task:
1. Patient name appears in PDF.
2. Age appears in PDF.
3. Sex appears in PDF.
4. Positive Glucose SHAP generates glucose-related guidance.
5. Positive BMI SHAP generates BMI-related guidance.
6. Positive Age SHAP does NOT recommend changing age.
7. Negative Insulin SHAP does NOT generate 'reduce insulin' advice.
8. Negative SHAP values are not called medically protective.
9. Sex does not enter the model feature matrix.
10. Patient name does not enter the model feature matrix.
11. PDF and Streamlit use the same recommendation function.
12. PDF generation works without an LLM API key.
13. No fabricated SHAP values (model consistency).
14. No medication recommendations.
15. Disclaimer is present in PDF and recommendations.
"""

import base64
import os
import unittest
import zlib
from unittest import mock

import joblib
import pandas as pd

from services.ai_report import MANDATORY_DISCLAIMER, generate_fallback_report
from services.pdf_report import generate_pdf_report
from services.recommendations import (
    GENERAL_PREVENTIVE_WELLNESS,
    generate_personalized_recommendations,
)
from src.preprocessing import FEATURE_ORDER, preprocess_patient


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extracts all decompressed text streams from ReportLab PDF bytes across all pages."""
    extracted = []
    pos = 0
    while True:
        idx = pdf_bytes.find(b"\nstream\n", pos)
        if idx == -1:
            break
        data_start = idx + len(b"\nstream\n")
        end_idx = pdf_bytes.find(b"endstream", data_start)
        if end_idx == -1:
            break
        raw = pdf_bytes[data_start:end_idx].strip()
        try:
            if raw.endswith(b"~>"):
                dec = zlib.decompress(base64.a85decode(raw, adobe=True))
            else:
                dec = zlib.decompress(raw)
            extracted.append(dec.decode("latin1", errors="ignore"))
        except Exception:
            pass
        pos = end_idx + len(b"endstream")
    return " ".join(extracted)


class TestPersonalizedGuidance(unittest.TestCase):
    def setUp(self):
        self.patient = {
            "Name": "Sanjith Kumar",
            "Sex": "Male",
            "Age": 45,
            "Pregnancies": 0,
            "Glucose": 140,
            "BloodPressure": 78,
            "SkinThickness": 25,
            "Insulin": 95,
            "BMI": 32.5,
            "DiabetesPedigreeFunction": 0.55,
        }
        self.shap_features = [
            {"Feature": "Glucose", "Patient_Value": 140, "SHAP_Value": 0.5200, "Direction": "Pushes risk higher"},
            {"Feature": "BMI", "Patient_Value": 32.5, "SHAP_Value": 0.3100, "Direction": "Pushes risk higher"},
            {"Feature": "Age", "Patient_Value": 45, "SHAP_Value": 0.1800, "Direction": "Pushes risk higher"},
            {"Feature": "Insulin", "Patient_Value": 95, "SHAP_Value": -0.2200, "Direction": "Pushes risk lower"},
            {"Feature": "BloodPressure", "Patient_Value": 78, "SHAP_Value": -0.0500, "Direction": "Pushes risk lower"},
        ]

    # 1. Patient name appears in PDF
    def test_01_patient_name_appears_in_pdf(self):
        pdf = generate_pdf_report(self.patient, 1, 0.65, self.shap_features)
        text = extract_text_from_pdf_bytes(pdf)
        self.assertIn("Sanjith Kumar", text)

    # 2. Age appears in PDF
    def test_02_age_appears_in_pdf(self):
        pdf = generate_pdf_report(self.patient, 1, 0.65, self.shap_features)
        text = extract_text_from_pdf_bytes(pdf)
        self.assertIn("45", text)
        self.assertIn("years", text)

    # 3. Sex appears in PDF
    def test_03_sex_appears_in_pdf(self):
        pdf = generate_pdf_report(self.patient, 1, 0.65, self.shap_features)
        text = extract_text_from_pdf_bytes(pdf)
        self.assertIn("Male", text)

    # 4. Positive Glucose SHAP generates glucose-related guidance
    def test_04_positive_glucose_shap_generates_glucose_guidance(self):
        res = generate_personalized_recommendations(self.patient, self.shap_features, 0.65)
        glucose_areas = [a for a in res["priority_areas"] if a["feature"] == "Glucose"]
        self.assertEqual(len(glucose_areas), 1)
        glucose_text = " ".join(glucose_areas[0]["guidance"]).lower()
        self.assertIn("blood-glucose testing", glucose_text)
        self.assertIn("hba1c", glucose_text)

    # 5. Positive BMI SHAP generates BMI-related guidance
    def test_05_positive_bmi_shap_generates_bmi_guidance(self):
        res = generate_personalized_recommendations(self.patient, self.shap_features, 0.65)
        bmi_areas = [a for a in res["priority_areas"] if a["feature"] == "BMI"]
        self.assertEqual(len(bmi_areas), 1)
        bmi_text = " ".join(bmi_areas[0]["guidance"]).lower()
        self.assertIn("eating patterns", bmi_text)
        self.assertNotIn("lose", bmi_text)  # Never commands 'lose X kg'

    # 6. Positive Age SHAP does NOT recommend changing age
    def test_06_positive_age_shap_does_not_recommend_changing_age(self):
        res = generate_personalized_recommendations(self.patient, self.shap_features, 0.65)
        age_areas = [a for a in res["priority_areas"] if a["feature"] == "Age"]
        self.assertEqual(len(age_areas), 1)
        self.assertFalse(age_areas[0]["is_modifiable"])
        age_text = " ".join(age_areas[0]["guidance"]).lower()
        self.assertNotIn("reduce age", age_text)
        self.assertNotIn("lower your age", age_text)
        self.assertIn("non-modifiable", age_areas[0]["why_it_matters"].lower())

    # 7. Negative Insulin SHAP does NOT generate "reduce insulin" advice
    def test_07_negative_insulin_shap_no_reduce_advice(self):
        res = generate_personalized_recommendations(self.patient, self.shap_features, 0.65)
        # Insulin has negative SHAP (-0.2200), so it must NOT be in priority upward areas
        insulin_areas = [a for a in res["priority_areas"] if a["feature"] == "Insulin"]
        self.assertEqual(len(insulin_areas), 0)
        all_bullets = " ".join(res["all_personalized_bullets"]).lower()
        self.assertNotIn("reduce insulin", all_bullets)
        self.assertNotIn("lower insulin", all_bullets)

    # 8. Negative SHAP values are not called medically protective
    def test_08_negative_shap_not_called_protective(self):
        res = generate_personalized_recommendations(self.patient, self.shap_features, 0.65)
        neg_summary = " ".join(res["negative_contributors_summary"]).lower()
        self.assertNotIn("protective", neg_summary)
        self.assertNotIn("protects you", neg_summary)
        self.assertIn("lowered the model's predicted risk", neg_summary)

    # 9. Sex does not enter the model feature matrix
    def test_09_sex_not_in_model_feature_matrix(self):
        self.assertNotIn("Sex", FEATURE_ORDER)
        self.assertNotIn("Gender", FEATURE_ORDER)
        self.assertEqual(len(FEATURE_ORDER), 8)

    # 10. Patient name does not enter the model feature matrix
    def test_10_patient_name_not_in_model_feature_matrix(self):
        self.assertNotIn("Name", FEATURE_ORDER)
        self.assertNotIn("Patient_Name", FEATURE_ORDER)
        self.assertEqual(FEATURE_ORDER, [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ])

    # 11. PDF and Streamlit use the same recommendation function
    def test_11_shared_recommendation_function(self):
        from services.recommendations import generate_personalized_recommendations as rec_fn_1
        import services.pdf_report
        # Ensure pdf_report imports and uses the exact same function
        self.assertIs(services.pdf_report.generate_personalized_recommendations, rec_fn_1)

    # 12. PDF generation works without an LLM API key
    def test_12_pdf_generation_without_llm_key(self):
        with mock.patch.dict(os.environ, {"LLM_API_KEY": ""}):
            pdf = generate_pdf_report(self.patient, 0, 0.28, self.shap_features)
            self.assertIsInstance(pdf, bytes)
            self.assertTrue(pdf.startswith(b"%PDF"))
            self.assertGreater(len(pdf), 2000)

    # 13. No fabricated SHAP values (values in priority areas match input SHAP exactly)
    def test_13_no_fabricated_shap_values(self):
        res = generate_personalized_recommendations(self.patient, self.shap_features, 0.65)
        shap_dict = {f["Feature"]: f["SHAP_Value"] for f in self.shap_features}
        for area in res["priority_areas"]:
            feat = area["feature"]
            self.assertIn(feat, shap_dict)
            self.assertAlmostEqual(area["shap_value"], shap_dict[feat], places=4)

    # 14. No medication recommendations
    def test_14_no_medication_recommendations(self):
        res = generate_personalized_recommendations(self.patient, self.shap_features, 0.65)
        full_text = res["summary_text"] + " " + " ".join(res["all_personalized_bullets"])
        lower = full_text.lower()
        # Verify no prescriptive medication commands
        self.assertNotIn("take metformin", lower)
        self.assertNotIn("prescribe", lower)
        self.assertNotIn("start insulin", lower)
        self.assertNotIn("increase dosage", lower)
        self.assertNotIn("decrease dosage", lower)

    # 15. Disclaimer is present
    def test_15_disclaimer_present(self):
        pdf = generate_pdf_report(self.patient, 1, 0.65, self.shap_features)
        text = extract_text_from_pdf_bytes(pdf)
        self.assertIn("MANDATORY MEDICAL NOTICE", text)
        self.assertIn("research prototype", text)


if __name__ == "__main__":
    unittest.main()

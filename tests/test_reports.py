"""Tests for AI report synthesis, 12-section grounding, and PDF document compilation."""

import os
import unittest
from unittest import mock
from services.ai_report import (
    generate_fallback_report,
    generate_patient_report,
    get_llm_status,
    MANDATORY_DISCLAIMER,
)
from services.pdf_report import generate_pdf_report


class TestReports(unittest.TestCase):
    def setUp(self):
        self.patient = {
            "Pregnancies": 1,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": 30.0,
            "DiabetesPedigreeFunction": 0.47,
            "Age": 33,
        }
        self.top_shap = [
            {"Feature": "Glucose", "Patient_Value": 120, "SHAP_Value": 0.45, "Direction": "Pushes risk higher"},
            {"Feature": "BMI", "Patient_Value": 30.0, "SHAP_Value": 0.25, "Direction": "Pushes risk higher"},
            {"Feature": "Age", "Patient_Value": 33, "SHAP_Value": -0.15, "Direction": "Pushes risk lower"},
        ]
        self.top_lime = [
            {"Condition": "Glucose > 115.0", "Feature": "Glucose", "LIME_Weight": 0.40, "Direction": "Pushes risk higher"},
            {"Condition": "BMI > 28.0", "Feature": "BMI", "LIME_Weight": 0.20, "Direction": "Pushes risk higher"},
        ]
        self.model_info = {"model_name": "XGBoost (Baseline)", "roc_auc": 0.8198}

    def test_get_llm_status(self):
        status = get_llm_status()
        self.assertIn("is_configured", status)
        self.assertIn("provider", status)
        self.assertIn("model", status)
        with mock.patch.dict(os.environ, {"LLM_API_KEY": ""}):
            unconfigured_status = get_llm_status()
            self.assertFalse(unconfigured_status["is_configured"])

    def test_12_section_fallback_report_structure(self):
        report = generate_fallback_report(
            self.patient, 0, 0.2286, self.top_shap, self.model_info, lime_features=self.top_lime
        )
        text = report["report_text"]
        self.assertIn("1. DIABETES RISK ASSESSMENT", text)
        self.assertIn("2. PREDICTION SUMMARY", text)
        self.assertIn("3. ESTIMATED PROBABILITY", text)
        self.assertIn("4. PATIENT INPUT SUMMARY", text)
        self.assertIn("5. KEY FACTORS INFLUENCING THE MODEL", text)
        self.assertIn("6. SHAP EXPLANATION", text)
        self.assertIn("7. LIME EXPLANATION", text)
        self.assertIn("8. MODEL INFORMATION", text)
        self.assertIn("9. INTERPRETATION OF THE PREDICTION", text)
        self.assertIn("10. GENERAL EDUCATIONAL INFORMATION", text)
        self.assertIn("11. LIMITATIONS", text)
        self.assertIn("12. MEDICAL DISCLAIMER", text)
        self.assertIn(MANDATORY_DISCLAIMER, text)

    def test_generate_patient_report_graceful_fallback(self):
        with mock.patch.dict(os.environ, {"LLM_API_KEY": ""}):
            report = generate_patient_report(
                self.patient, 0, 0.2286, self.top_shap, self.model_info, lime_results=self.top_lime
            )
            self.assertIsInstance(report, dict)
            self.assertIn("disclaimer", report)
            self.assertIn("interpretation", report)
            self.assertFalse(report["is_ai_generated"])
            self.assertTrue(len(report["recommendations"]) >= 3)

    def test_pdf_report_compilation(self):
        report = generate_fallback_report(
            self.patient, 0, 0.2286, self.top_shap, self.model_info
        )
        pdf_bytes = generate_pdf_report(
            patient_data=self.patient,
            prediction=0,
            probability=0.2286,
            top_shap_features=self.top_shap,
            model_info=self.model_info,
            interpretation_text=report["interpretation"],
            recommendations=report["recommendations"],
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()

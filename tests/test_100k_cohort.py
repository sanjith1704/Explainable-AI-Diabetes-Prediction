"""
tests/test_100k_cohort.py

Automated test suite verifying the 100,000-record Clinical Cohort pipeline:
- Ingestion and leak-free preprocessing
- Feature scaling and patient serialization
- XGBoost & Calibrated XGBoost inference
- Tree SHAP local attributions
- Deterministic personalized clinical recommendations
- 9-Section PDF report compilation
"""

from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
import shap

from services.pdf_report import generate_pdf_report
from services.recommendations import generate_personalized_recommendations
from src.preprocessing_100k import (
    DISPLAY_LABELS_100K,
    FEATURE_ORDER_100K,
    NUMERICAL_FEATURES_100K,
    load_raw_100k_data,
    preprocess_patient_100k,
    transform_100k_features,
)
from src.shap_analysis import explain_single_patient


class Test100kCohort(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = Path("models")
        cls.model_raw = joblib.load(cls.models_dir / "xgboost_100k.pkl")
        cls.model_calibrated = joblib.load(cls.models_dir / "xgboost_100k_calibrated.pkl")
        cls.scaler = joblib.load(cls.models_dir / "scaler_100k.pkl")
        cls.tree_explainer = shap.TreeExplainer(cls.model_raw)

    def test_01_dataset_integrity(self):
        """Verify the 100k dataset loads cleanly and adheres to schema."""
        raw_df = load_raw_100k_data("data/diabetes_prediction_dataset.csv")
        self.assertGreater(len(raw_df), 90000, "Cleaned dataset should exceed 90,000 unique records.")
        for col in FEATURE_ORDER_100K:
            self.assertIn(col, raw_df.columns)
        self.assertIn("diabetes", raw_df.columns)
        self.assertEqual(raw_df.isnull().sum().sum(), 0, "Cleaned dataset should contain zero null values.")

    def test_02_patient_preprocessing(self):
        """Verify single patient serialization and scaling produces correct shape and columns."""
        sample_patient = {
            "gender": "Male",
            "age": 52,
            "hypertension": 1,
            "heart_disease": 0,
            "smoking_history": "former",
            "bmi": 31.4,
            "HbA1c_level": 7.1,
            "blood_glucose_level": 175,
        }
        scaled_df = preprocess_patient_100k(sample_patient, self.scaler)
        self.assertEqual(scaled_df.shape, (1, 8))
        self.assertEqual(list(scaled_df.columns), FEATURE_ORDER_100K)
        # Binary features should remain unscaled
        self.assertEqual(scaled_df["gender"].iloc[0], 1)
        self.assertEqual(scaled_df["hypertension"].iloc[0], 1)
        self.assertEqual(scaled_df["heart_disease"].iloc[0], 0)

    def test_03_model_inference_probabilities(self):
        """Verify raw and calibrated models generate valid probabilities in [0, 1]."""
        high_risk_patient = {
            "gender": "Female",
            "age": 65,
            "hypertension": 1,
            "heart_disease": 1,
            "smoking_history": "current",
            "bmi": 36.2,
            "HbA1c_level": 8.5,
            "blood_glucose_level": 240,
        }
        scaled_df = preprocess_patient_100k(high_risk_patient, self.scaler)
        prob_raw = float(self.model_raw.predict_proba(scaled_df)[0, 1])
        prob_cal = float(self.model_calibrated.predict_proba(scaled_df)[0, 1])

        self.assertGreaterEqual(prob_raw, 0.0)
        self.assertLessEqual(prob_raw, 1.0)
        self.assertGreaterEqual(prob_cal, 0.0)
        self.assertLessEqual(prob_cal, 1.0)
        # Severely elevated HbA1c and glucose should trigger high probability
        self.assertGreater(prob_raw, 0.80)
        self.assertGreater(prob_cal, 0.80)

    def test_04_tree_shap_local_attributions(self):
        """Verify SHAP produces complete local attributions for all 8 features."""
        patient = {
            "gender": "Male",
            "age": 45,
            "hypertension": 0,
            "heart_disease": 0,
            "smoking_history": "never",
            "bmi": 24.5,
            "HbA1c_level": 5.4,
            "blood_glucose_level": 95,
        }
        scaled_df = preprocess_patient_100k(patient, self.scaler)
        shap_df = explain_single_patient(self.tree_explainer, scaled_df, patient, features=FEATURE_ORDER_100K)

        self.assertEqual(len(shap_df), 8)
        self.assertIn("Feature", shap_df.columns)
        self.assertIn("SHAP_Value", shap_df.columns)
        self.assertIn("Direction", shap_df.columns)
        # Normal HbA1c should lower risk
        hba1c_row = shap_df[shap_df["Feature"] == "HbA1c_level"].iloc[0]
        self.assertLess(hba1c_row["SHAP_Value"], 0.0, "Normal HbA1c (5.4%) should have negative SHAP attribution.")

    def test_05_personalized_recommendations_100k(self):
        """Verify personalized guidance engine produces ADA criteria for elevated HbA1c."""
        patient = {
            "Name": "Alex Rivera",
            "Sex": "Male",
            "gender": "Male",
            "age": 55,
            "hypertension": 1,
            "heart_disease": 0,
            "smoking_history": "current",
            "bmi": 32.0,
            "HbA1c_level": 7.4,
            "blood_glucose_level": 180,
        }
        scaled_df = preprocess_patient_100k(patient, self.scaler)
        shap_df = explain_single_patient(self.tree_explainer, scaled_df, patient, features=FEATURE_ORDER_100K)

        guidance = generate_personalized_recommendations(
            patient_data=patient,
            shap_features=shap_df.to_dict(orient="records"),
            prediction_probability=0.88,
        )

        self.assertIn("summary_text", guidance)
        self.assertIn("priority_areas", guidance)
        # Verify HbA1c guidance is generated
        priority_features = [p["feature"] for p in guidance["priority_areas"]]
        self.assertIn("HbA1c_level", priority_features)

        # Check for ADA criteria in HbA1c priority guidance
        hba1c_guidance = [p for p in guidance["priority_areas"] if p["feature"] == "HbA1c_level"][0]
        full_text = " ".join(hba1c_guidance["guidance"]).lower()
        self.assertIn("ada", full_text)
        self.assertIn("5.7%", full_text)
        self.assertIn("6.5%", full_text)

    def test_06_pdf_report_compilation_100k(self):
        """Verify 9-section PDF report builds successfully for a 100k cohort patient."""
        patient = {
            "Name": "Carlos Mendoza",
            "Sex": "Male",
            "gender": "Male",
            "age": 60,
            "hypertension": 1,
            "heart_disease": 1,
            "smoking_history": "former",
            "bmi": 29.8,
            "HbA1c_level": 6.8,
            "blood_glucose_level": 160,
        }
        scaled_df = preprocess_patient_100k(patient, self.scaler)
        shap_df = explain_single_patient(self.tree_explainer, scaled_df, patient, features=FEATURE_ORDER_100K)
        guidance = generate_personalized_recommendations(
            patient_data=patient,
            shap_features=shap_df.to_dict(orient="records"),
            prediction_probability=0.78,
        )

        model_info = {"model_name": "Calibrated XGBoost (100k Cohort)", "roc_auc": 0.9770}
        pdf_bytes = generate_pdf_report(
            patient_data=patient,
            prediction=1,
            probability=0.78,
            top_shap_features=shap_df.to_dict(orient="records"),
            model_info=model_info,
            interpretation_text="Elevated metabolic risk assessment driven by HbA1c and systolic load.",
            recommendations=guidance["all_personalized_bullets"],
            personalized_guidance=guidance,
        )

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 5000, "Compiled PDF must be larger than 5KB.")
        self.assertTrue(pdf_bytes.startswith(b"%PDF"), "PDF stream must begin with standard %PDF magic header.")


if __name__ == "__main__":
    unittest.main()

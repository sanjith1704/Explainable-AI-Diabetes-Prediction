"""Tests for SHAP, LIME, agreement, sensitivity, and robustness workflows."""

import unittest
import joblib
import pandas as pd
from src.preprocessing import FEATURE_ORDER, preprocess_patient
from src.shap_analysis import create_tree_explainer, explain_single_patient
from src.lime_analysis import create_lime_explainer, explain_patient_lime
from src.agreement import compare_shap_and_lime
from src.faithfulness import compute_feature_sensitivity
from src.stability import evaluate_prediction_robustness


class TestExplainability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load("models/xgboost.pkl")
        cls.scaler = joblib.load("models/scaler.pkl")
        cls.imputer = joblib.load("models/imputer.pkl")
        cls.X_train = pd.read_csv("data/X_train.csv")

        cls.tree_explainer = create_tree_explainer(cls.model)
        cls.lime_explainer = create_lime_explainer(cls.X_train, random_state=42)

        cls.patient = {
            "Pregnancies": 1,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": 30.0,
            "DiabetesPedigreeFunction": 0.47,
            "Age": 33,
        }
        df = pd.DataFrame([cls.patient], columns=FEATURE_ORDER)
        cls.patient_scaled = preprocess_patient(df, cls.imputer, cls.scaler)

    def test_local_shap_explanation(self):
        shap_df = explain_single_patient(self.tree_explainer, self.patient_scaled, self.patient)
        self.assertEqual(len(shap_df), 8)
        self.assertIn("SHAP_Value", shap_df.columns)
        self.assertIn("Direction", shap_df.columns)

    def test_local_lime_explanation(self):
        lime_df = explain_patient_lime(
            self.lime_explainer, self.model.predict_proba, self.patient_scaled.values[0]
        )
        self.assertEqual(len(lime_df), 8)
        self.assertIn("LIME_Weight", lime_df.columns)

    def test_shap_lime_agreement(self):
        shap_df = explain_single_patient(self.tree_explainer, self.patient_scaled, self.patient)
        lime_df = explain_patient_lime(
            self.lime_explainer, self.model.predict_proba, self.patient_scaled.values[0]
        )
        _, metrics = compare_shap_and_lime(shap_df, lime_df)
        self.assertIn("spearman_rank_correlation", metrics)
        self.assertIn("jaccard_top3_overlap", metrics)
        self.assertGreater(metrics["jaccard_top3_overlap"], 0.0)

    def test_sensitivity_sweep(self):
        sens_dict = compute_feature_sensitivity(self.patient, self.model, self.imputer, self.scaler)
        self.assertIn("Glucose", sens_dict)
        self.assertIn("BMI", sens_dict)
        glucose_df = sens_dict["Glucose"]
        self.assertGreater(len(glucose_df), 10)
        self.assertTrue((glucose_df["Predicted_Probability"] >= 0).all())
        self.assertTrue((glucose_df["Predicted_Probability"] <= 1).all())

    def test_robustness_perturbation(self):
        _, metrics = evaluate_prediction_robustness(
            self.patient, self.model, self.imputer, self.scaler, n_perturbations=10, noise_level=0.01
        )
        self.assertIn("robustness_index", metrics)
        self.assertGreaterEqual(metrics["robustness_index"], 0.0)
        self.assertLessEqual(metrics["robustness_index"], 1.0)


if __name__ == "__main__":
    unittest.main()

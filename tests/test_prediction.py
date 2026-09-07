"""Tests for preprocessing, prediction, and probability generation."""

import unittest
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import FEATURE_ORDER, preprocess_patient


class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.model = joblib.load("models/xgboost.pkl")
        self.scaler = joblib.load("models/scaler.pkl")
        self.imputer = joblib.load("models/imputer.pkl")

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

    def test_prediction_output_and_bounds(self):
        df = pd.DataFrame([self.patient], columns=FEATURE_ORDER)
        processed = preprocess_patient(df, self.imputer, self.scaler)

        pred = self.model.predict(processed)[0]
        prob = self.model.predict_proba(processed)[0, 1]

        self.assertIn(pred, [0, 1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        # Verify baseline probability for reference patient (~22.86%)
        self.assertAlmostEqual(prob, 0.228597, places=4)

    def test_missing_zero_insulin_imputation(self):
        patient_zero = self.patient.copy()
        patient_zero["Insulin"] = 0  # Missing-style zero

        df = pd.DataFrame([patient_zero], columns=FEATURE_ORDER)
        processed = preprocess_patient(df, self.imputer, self.scaler)

        prob = self.model.predict_proba(processed)[0, 1]
        # Verify imputed zero-insulin probability reproduces ~42.00%
        self.assertAlmostEqual(prob, 0.420033, places=4)

    def test_clinical_feature_engineer(self):
        from src.preprocessing import ClinicalFeatureEngineer
        fe = ClinicalFeatureEngineer(
            include_glucose_bmi=True,
            include_insulin_glucose_ratio=True,
            include_clinical_cutoffs=True,
            include_age_pregnancies=True,
            include_binned=True,
        )
        df = pd.DataFrame([self.patient], columns=FEATURE_ORDER)
        fe.fit(df)
        transformed = fe.transform(df)
        self.assertIn("Glucose_BMI_interaction", transformed.columns)
        self.assertIn("Insulin_Glucose_ratio", transformed.columns)
        self.assertIn("Glucose_over_126", transformed.columns)
        self.assertEqual(transformed["Glucose_BMI_interaction"].iloc[0], 120.0 * 30.0)

    def test_resampling_smote_and_adasyn(self):
        from src.preprocessing import smote_oversample, adasyn_oversample
        X_toy = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [5.0, 5.0], [5.1, 5.1], [5.2, 5.2], [5.3, 5.3], [5.4, 5.4]])
        y_toy = np.array([1, 1, 1, 0, 0, 0, 0, 0])

        X_smote, y_smote = smote_oversample(X_toy, y_toy, random_state=42, k_neighbors=2)
        self.assertEqual(len(y_smote), 10)  # 5 class 0, 5 class 1
        self.assertEqual(np.sum(y_smote == 1), 5)

        X_ada, y_ada = adasyn_oversample(X_toy, y_toy, random_state=42, k_neighbors=2)
        self.assertGreaterEqual(len(y_ada), 8)
        self.assertGreaterEqual(np.sum(y_ada == 1), 3)

    def test_threshold_analysis_includes_f2(self):
        from src.evaluation import compute_threshold_analysis
        y_true = pd.Series([1, 0, 1, 1, 0, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.7])
        df_th = compute_threshold_analysis(y_true, y_prob, thresholds=[0.5])
        self.assertIn("F2 Score", df_th.columns)
        self.assertIn("Recall (Sensitivity)", df_th.columns)
        self.assertGreater(df_th["F2 Score"].iloc[0], 0.0)


if __name__ == "__main__":
    unittest.main()

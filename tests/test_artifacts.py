"""Tests for model and preprocessor serialization artifacts."""

import unittest
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class TestArtifacts(unittest.TestCase):
    def test_baseline_xgboost_exists_and_loads(self):
        path = Path("models/xgboost.pkl")
        self.assertTrue(path.exists(), "models/xgboost.pkl does not exist")
        model = joblib.load(path)
        self.assertIsInstance(model, XGBClassifier)
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.max_depth, 3)

    def test_optimized_xgboost_exists_and_loads(self):
        path = Path("models/xgboost_optimized.pkl")
        self.assertTrue(path.exists(), "models/xgboost_optimized.pkl does not exist")
        model = joblib.load(path)
        self.assertIsInstance(model, XGBClassifier)

    def test_scaler_exists_and_loads(self):
        path = Path("models/scaler.pkl")
        self.assertTrue(path.exists(), "models/scaler.pkl does not exist")
        scaler = joblib.load(path)
        self.assertIsInstance(scaler, StandardScaler)
        self.assertEqual(len(scaler.mean_), 8)

    def test_imputer_exists_and_loads(self):
        path = Path("models/imputer.pkl")
        self.assertTrue(path.exists(), "models/imputer.pkl does not exist")
        imputer = joblib.load(path)
        self.assertIsInstance(imputer, SimpleImputer)
        self.assertEqual(len(imputer.statistics_), 8)


if __name__ == "__main__":
    unittest.main()

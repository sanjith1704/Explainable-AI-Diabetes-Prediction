"""Tests for input validation rules and boundary checking."""

import unittest
from services.validation import validate_patient_inputs


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.valid_patient = {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 75,
            "SkinThickness": 25,
            "Insulin": 100,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.55,
            "Age": 35,
        }

    def test_valid_patient_passes(self):
        is_valid, errors = validate_patient_inputs(self.valid_patient)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_negative_pregnancies_rejected(self):
        patient = self.valid_patient.copy()
        patient["Pregnancies"] = -1
        is_valid, errors = validate_patient_inputs(patient)
        self.assertFalse(is_valid)
        self.assertTrue(any("Pregnancies" in e for e in errors))

    def test_negative_glucose_rejected(self):
        patient = self.valid_patient.copy()
        patient["Glucose"] = -5
        is_valid, errors = validate_patient_inputs(patient)
        self.assertFalse(is_valid)
        self.assertTrue(any("Glucose" in e for e in errors))

    def test_extreme_bmi_rejected(self):
        patient = self.valid_patient.copy()
        patient["BMI"] = 120.0
        is_valid, errors = validate_patient_inputs(patient)
        self.assertFalse(is_valid)
        self.assertTrue(any("BMI" in e for e in errors))

    def test_underage_rejected(self):
        patient = self.valid_patient.copy()
        patient["Age"] = 15  # Minimum age 18 for PIMA adult cohort
        is_valid, errors = validate_patient_inputs(patient)
        self.assertFalse(is_valid)
        self.assertTrue(any("Age" in e for e in errors))

    def test_missing_field_rejected(self):
        patient = self.valid_patient.copy()
        del patient["Insulin"]
        is_valid, errors = validate_patient_inputs(patient)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required" in e for e in errors))


if __name__ == "__main__":
    unittest.main()

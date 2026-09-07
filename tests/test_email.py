"""Tests for email validation, Part 8 formatting, and SMTP handling."""

import unittest
from services.email_service import (
    check_email_configuration,
    format_email_body,
    send_report_email,
    send_test_email,
    validate_email_address,
)


class TestEmailService(unittest.TestCase):
    def test_email_validation_valid(self):
        self.assertTrue(validate_email_address("doctor@hospital.org"))
        self.assertTrue(validate_email_address("researcher.jane@university.edu"))
        self.assertTrue(validate_email_address("user+tag@gmail.com"))

    def test_email_validation_invalid(self):
        self.assertFalse(validate_email_address("notanemail"))
        self.assertFalse(validate_email_address("missing@domain"))
        self.assertFalse(validate_email_address("@nodomain.com"))
        self.assertFalse(validate_email_address(""))
        self.assertFalse(validate_email_address(None))

    def test_send_email_rejects_invalid_address(self):
        success, msg = send_report_email("bad-email", "Subject", "Body")
        self.assertFalse(success)
        self.assertIn("valid email address", msg.lower())

    def test_send_email_graceful_handling_unconfigured(self):
        # In testing environment without real SMTP, should fail gracefully without crashing
        success, msg = send_report_email("valid@example.com", "Subject", "Body")
        self.assertIsInstance(success, bool)
        self.assertIsInstance(msg, str)

    def test_format_email_body_structure(self):
        top_shap = [
            {"Feature": "Glucose", "SHAP_Value": 0.45, "Direction": "Pushes risk higher"},
            {"Feature": "BMI", "SHAP_Value": 0.25, "Direction": "Pushes risk higher"},
        ]
        body = format_email_body(
            prediction=1,
            probability=0.75,
            top_shap_features=top_shap,
            interpretation_text="The model estimated elevated risk.",
            model_name="XGBoost (Baseline)",
        )
        self.assertIn("Diabetes Risk Assessment", body)
        self.assertIn("Prediction:\nHigher Risk (Class 1)", body)
        self.assertIn("Estimated Probability:\n75.00%", body)
        self.assertIn("Key Model Factors:", body)
        self.assertIn("Glucose — +0.4500", body)
        self.assertIn("AI Interpretation:", body)
        self.assertIn("Model:\nXGBoost (Baseline)", body)
        self.assertIn("Disclaimer:", body)

    def test_send_test_email_rejects_invalid(self):
        success, msg = send_test_email("invalid-address")
        self.assertFalse(success)
        self.assertIn("valid email address", msg.lower())


if __name__ == "__main__":
    unittest.main()

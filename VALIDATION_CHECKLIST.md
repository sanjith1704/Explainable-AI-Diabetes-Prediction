# Validation checklist

Use this checklist before submission, demo, or viva.

- [x] Confirm the deployed app uses the saved `imputer.pkl`, `scaler.pkl`, and `xgboost.pkl`.
- [x] Confirm the feature order remains `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
- [x] Confirm UI labels include units but the internal DataFrame column names remain unchanged.
- [x] Confirm a non-zero sample patient produces the same prediction in the app and in `scripts/validate_deployed_artifacts.py`.
- [x] Confirm missing-style zero values for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are imputed before scaling.
- [x] Confirm the deployed holdout metrics reproduce approximately `74.03%`, `65.22%`, `55.56%`, `60.00%`, and `81.98%`.
- [x] Confirm SHAP explanations are generated from the same processed input that is sent to XGBoost.
- [x] Confirm SHAP is described as a model explanation, not medical causation.
- [x] Confirm the app shows an educational and non-diagnostic disclaimer.
- [x] Confirm the project documentation does not claim clinical validation.
- [ ] Optional future cleanup: repair notebook cells with stale metrics, double scaling, and test-only cross-validation.

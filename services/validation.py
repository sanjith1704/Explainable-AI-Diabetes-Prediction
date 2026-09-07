"""
validation.py: Input validation and domain boundary enforcement for clinical parameters.
Rejects invalid physiological inputs with descriptive, user-friendly guidance.
"""

from typing import Any, Dict, List, Tuple

VALIDATION_RULES = {
    "Pregnancies": {
        "min": 0,
        "max": 20,
        "type": int,
        "label": "Pregnancies",
        "description": "Must be an integer between 0 and 20.",
    },
    "Glucose": {
        "min": 0,
        "max": 300,
        "type": (int, float),
        "label": "Glucose",
        "description": "Plasma glucose concentration must be between 0 and 300 mg/dL (0 indicates unrecorded/missing).",
    },
    "BloodPressure": {
        "min": 0,
        "max": 200,
        "type": (int, float),
        "label": "Blood Pressure",
        "description": "Diastolic blood pressure must be between 0 and 200 mmHg (0 indicates unrecorded/missing).",
    },
    "SkinThickness": {
        "min": 0,
        "max": 100,
        "type": (int, float),
        "label": "Skin Thickness",
        "description": "Triceps skinfold thickness must be between 0 and 100 mm (0 indicates unrecorded/missing).",
    },
    "Insulin": {
        "min": 0,
        "max": 900,
        "type": (int, float),
        "label": "Insulin",
        "description": "2-hour serum insulin must be between 0 and 900 muU/mL (0 indicates unrecorded/missing).",
    },
    "BMI": {
        "min": 0.0,
        "max": 75.0,
        "type": (int, float),
        "label": "Body Mass Index (BMI)",
        "description": "BMI must be between 0.0 and 75.0 kg/m^2 (0 indicates unrecorded/missing).",
    },
    "DiabetesPedigreeFunction": {
        "min": 0.05,
        "max": 3.0,
        "type": (int, float),
        "label": "Diabetes Pedigree Function",
        "description": "Diabetes Pedigree Function must be a positive ratio between 0.05 and 3.00.",
    },
    "Age": {
        "min": 18,
        "max": 120,
        "type": int,
        "label": "Age",
        "description": "Adult patient age must be between 18 and 120 years.",
    },
}


def validate_patient_inputs(inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates a complete set of 8 patient features against clinical domain bounds.
    Returns:
        is_valid: True if all features conform to valid bounds, False otherwise.
        errors: List of friendly validation error messages explaining rejected values.
    """
    errors: List[str] = []

    for feature_name, rule in VALIDATION_RULES.items():
        if feature_name not in inputs:
            errors.append(f"Missing required clinical parameter: '{rule['label']}'.")
            continue

        raw_val = inputs[feature_name]
        # Check numerical type
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            errors.append(f"Invalid input for '{rule['label']}': value '{raw_val}' is not a valid number.")
            continue

        # Check domain boundaries
        if val < rule["min"]:
            errors.append(
                f"'{rule['label']}' cannot be negative or below minimum threshold {rule['min']} (entered: {val})."
            )
        elif val > rule["max"]:
            errors.append(
                f"'{rule['label']}' exceeds plausible maximum threshold {rule['max']} (entered: {val})."
            )

    is_valid = len(errors) == 0
    return is_valid, errors

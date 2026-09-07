"""
stability.py: Prediction stability and local explanation robustness testing.
Evaluates model behavior under minor physiological perturbations to ensure
predictions and explanations do not exhibit erratic discontinuities.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.preprocessing import FEATURE_ORDER, preprocess_patient


def evaluate_prediction_robustness(
    patient_dict: Dict[str, Any],
    model: Any,
    imputer: Any,
    scaler: Any,
    n_perturbations: int = 50,
    noise_level: float = 0.02,  # 2% relative perturbation
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simulates minor measurement variance around patient input features.
    Computes probability distribution, prediction flip rate, and stability index.
    """
    rng = np.random.default_rng(random_state)
    base_df = pd.DataFrame([patient_dict], columns=FEATURE_ORDER)
    base_processed = preprocess_patient(base_df, imputer, scaler)
    base_prob = float(model.predict_proba(base_processed)[0, 1])
    base_pred = int(model.predict(base_processed)[0])

    perturbed_records = []
    probabilities = []
    classes = []

    for i in range(n_perturbations):
        perturbed = patient_dict.copy()
        for feat in FEATURE_ORDER:
            orig_val = float(patient_dict[feat])
            if orig_val > 0:
                # Add Gaussian noise with standard deviation proportional to noise_level
                delta = rng.normal(0, noise_level * orig_val)
                # Keep values non-negative
                new_val = max(0.0, orig_val + delta)
                perturbed[feat] = round(new_val, 2)

        pert_df = pd.DataFrame([perturbed], columns=FEATURE_ORDER)
        pert_processed = preprocess_patient(pert_df, imputer, scaler)
        pert_prob = float(model.predict_proba(pert_processed)[0, 1])
        pert_pred = int(model.predict(pert_processed)[0])

        probabilities.append(pert_prob)
        classes.append(pert_pred)
        perturbed["Iteration"] = i + 1
        perturbed["Perturbed_Probability"] = round(pert_prob, 4)
        perturbed["Perturbed_Prediction"] = pert_pred
        perturbed["Absolute_Shift"] = round(abs(pert_prob - base_prob), 4)
        perturbed_records.append(perturbed)

    pert_results_df = pd.DataFrame(perturbed_records)

    prob_arr = np.array(probabilities)
    class_arr = np.array(classes)

    mean_prob = float(np.mean(prob_arr))
    std_prob = float(np.std(prob_arr))
    max_shift = float(np.max(np.abs(prob_arr - base_prob)))
    flip_rate = float(np.mean(class_arr != base_pred))

    # Robustness index: bounded [0, 1] where 1.0 indicates absolute stability
    robustness_index = float(max(0.0, 1.0 - (std_prob / 0.25)))

    metrics = {
        "base_probability": base_prob,
        "base_prediction": base_pred,
        "mean_perturbed_probability": mean_prob,
        "std_perturbed_probability": std_prob,
        "max_probability_shift": max_shift,
        "prediction_flip_rate": flip_rate,
        "robustness_index": robustness_index,
        "samples_evaluated": n_perturbations,
    }

    return pert_results_df, metrics

"""
preprocessing.py: Modular data preprocessing and pipeline definitions for diabetes prediction.
Preserves baseline pipeline reproducibility while providing improved missing-value handling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_ORDER: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Features where 0 represents an unrecorded / missing physiological measurement
INVALID_ZERO_FEATURES: List[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# Physiological rationale for zero-value imputation
ZERO_VALUE_RATIONALE: Dict[str, str] = {
    "Glucose": "Fasting plasma glucose cannot be 0 mg/dL in a living individual (hypoglycemic coma occurs well above 0).",
    "BloodPressure": "Diastolic blood pressure of 0 mmHg indicates circulatory arrest / measurement absence.",
    "SkinThickness": "Triceps skinfold thickness cannot physiologically be 0 mm.",
    "Insulin": "Two-hour serum insulin of 0 muU/mL indicates absence of laboratory assay or unrecorded test.",
    "BMI": "Body mass index cannot biologically be 0 kg/m^2.",
    "Pregnancies": "Zero pregnancies is clinically valid and common (nulliparous individuals); therefore 0 is preserved.",
    "DiabetesPedigreeFunction": "Score cannot biologically be 0 and has no zero entries in the PIMA dataset.",
    "Age": "Age is positive in all records with no zero entries in the PIMA dataset.",
}


class MissingZeroHandler(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to convert biologically implausible zeros
    to NaN values for specified physiological features.
    """

    def __init__(self, features: Optional[List[str]] = None):
        self.features = features or INVALID_ZERO_FEATURES

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=FEATURE_ORDER)
        for col in self.features:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
        return df


def load_raw_data(data_path: Union[str, Path] = "data/diabetes.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """Load raw PIMA dataset and separate features from outcome."""
    df = pd.read_csv(data_path)
    X = df[FEATURE_ORDER].copy()
    y = df["Outcome"].astype(int).copy()
    return X, y


def build_baseline_pipeline(model: BaseEstimator) -> Pipeline:
    """
    Builds the baseline pipeline:
    1. Zero-to-NaN conversion for invalid features
    2. Median imputation (SimpleImputer)
    3. Standard scaling (StandardScaler)
    4. Estimator
    """
    return Pipeline(
        [
            ("zero_handler", MissingZeroHandler(features=INVALID_ZERO_FEATURES)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_preprocessor_pipeline() -> Pipeline:
    """Builds the standalone preprocessing pipeline (zero handler + imputer + scaler)."""
    return Pipeline(
        [
            ("zero_handler", MissingZeroHandler(features=INVALID_ZERO_FEATURES)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def preprocess_patient(
    patient_df: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Preprocess a single patient or batch using pre-fitted imputer and scaler artifacts,
    strictly replicating the deployed Streamlit inference flow.
    """
    cleaned = patient_df.copy()
    for col in INVALID_ZERO_FEATURES:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].replace(0, np.nan)

    imputed_arr = imputer.transform(cleaned[FEATURE_ORDER])
    imputed_df = pd.DataFrame(imputed_arr, columns=FEATURE_ORDER, index=patient_df.index)

    scaled_arr = scaler.transform(imputed_df.values)
    scaled_df = pd.DataFrame(scaled_arr, columns=FEATURE_ORDER, index=patient_df.index)
    return scaled_df


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Leak-free clinical feature engineering transformer for diabetes risk prediction.
    Features:
    - Glucose_BMI_interaction: Glucose * BMI (metabolic risk synergy)
    - Insulin_Glucose_ratio: Insulin / (Glucose + 1e-5) (beta cell function surrogate)
    - Glucose_over_126: ADA fasting plasma glucose diagnostic cutoff (>=126 mg/dL)
    - BMI_over_30: WHO obesity cutoff (>=30 kg/m^2)
    - Age_over_45: ADA diabetes screening age threshold (>=45 years)
    - Age_Pregnancies_interaction: Age * Pregnancies (gestational cumulative risk)
    - Glucose_binned, BMI_binned: Quantile-based ordinal bins (fit strictly on training data)
    """

    def __init__(
        self,
        include_glucose_bmi: bool = True,
        include_insulin_glucose_ratio: bool = True,
        include_clinical_cutoffs: bool = True,
        include_age_pregnancies: bool = True,
        include_binned: bool = False,
    ):
        self.include_glucose_bmi = include_glucose_bmi
        self.include_insulin_glucose_ratio = include_insulin_glucose_ratio
        self.include_clinical_cutoffs = include_clinical_cutoffs
        self.include_age_pregnancies = include_age_pregnancies
        self.include_binned = include_binned
        self.glucose_bins_: Optional[np.ndarray] = None
        self.bmi_bins_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = list(FEATURE_ORDER)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=FEATURE_ORDER[: X.shape[1]])

        if self.include_binned:
            # Fit quantiles strictly on training data (quartiles: 25%, 50%, 75%)
            if "Glucose" in df.columns:
                self.glucose_bins_ = np.percentile(df["Glucose"].dropna(), [25, 50, 75])
            if "BMI" in df.columns:
                self.bmi_bins_ = np.percentile(df["BMI"].dropna(), [25, 50, 75])

        # Compute output feature names
        names = list(FEATURE_ORDER)
        if self.include_glucose_bmi:
            names.append("Glucose_BMI_interaction")
        if self.include_insulin_glucose_ratio:
            names.append("Insulin_Glucose_ratio")
        if self.include_clinical_cutoffs:
            names.extend(["Glucose_over_126", "BMI_over_30", "Age_over_45"])
        if self.include_age_pregnancies:
            names.append("Age_Pregnancies_interaction")
        if self.include_binned:
            names.extend(["Glucose_binned", "BMI_binned"])
        self.feature_names_ = names
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X[FEATURE_ORDER].copy()
        else:
            df = pd.DataFrame(X, columns=FEATURE_ORDER[: X.shape[1]])

        if self.include_glucose_bmi:
            df["Glucose_BMI_interaction"] = df["Glucose"] * df["BMI"]

        if self.include_insulin_glucose_ratio:
            df["Insulin_Glucose_ratio"] = df["Insulin"] / (df["Glucose"] + 1e-5)

        if self.include_clinical_cutoffs:
            df["Glucose_over_126"] = (df["Glucose"] >= 126.0).astype(float)
            df["BMI_over_30"] = (df["BMI"] >= 30.0).astype(float)
            df["Age_over_45"] = (df["Age"] >= 45.0).astype(float)

        if self.include_age_pregnancies:
            df["Age_Pregnancies_interaction"] = df["Age"] * df["Pregnancies"]

        if self.include_binned:
            if self.glucose_bins_ is not None:
                df["Glucose_binned"] = np.digitize(df["Glucose"], self.glucose_bins_).astype(float)
            if self.bmi_bins_ is not None:
                df["BMI_binned"] = np.digitize(df["BMI"], self.bmi_bins_).astype(float)

        return df

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return list(self.feature_names_)


def smote_oversample(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    random_state: int = 42,
    k_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic Minority Over-sampling Technique (SMOTE; Chawla et al. 2002).
    Synthesizes minority instances by linear interpolation between nearest minority neighbors.
    Must be called ONLY on training folds to prevent data leakage.
    """
    rng = np.random.RandomState(random_state)
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)

    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2:
        return X_arr, y_arr

    if counts[0] >= counts[1]:
        maj_class, min_class = classes[0], classes[1]
        n_maj, n_min = counts[0], counts[1]
    else:
        maj_class, min_class = classes[1], classes[0]
        n_maj, n_min = counts[1], counts[0]

    n_to_generate = n_maj - n_min
    if n_to_generate <= 0:
        return X_arr, y_arr

    X_min = X_arr[y_arr == min_class]
    k = min(k_neighbors, len(X_min) - 1)
    if k < 1:
        return X_arr, y_arr

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_min)
    indices = nn.kneighbors(X_min, return_distance=False)

    base_indices = rng.choice(len(X_min), size=n_to_generate, replace=True)
    neighbor_offsets = rng.randint(1, k + 1, size=n_to_generate)
    lambdas = rng.uniform(0.0, 1.0, size=(n_to_generate, 1))

    synth_samples = np.empty((n_to_generate, X_arr.shape[1]), dtype=float)
    for idx in range(n_to_generate):
        base_idx = base_indices[idx]
        nbr_col = neighbor_offsets[idx]
        nbr_idx = indices[base_idx, nbr_col]
        x_base = X_min[base_idx]
        x_nbr = X_min[nbr_idx]
        diff = x_nbr - x_base
        synth_samples[idx] = x_base + lambdas[idx] * diff

    synth_labels = np.full(n_to_generate, min_class, dtype=int)
    X_resampled = np.vstack([X_arr, synth_samples])
    y_resampled = np.concatenate([y_arr, synth_labels])

    perm = rng.permutation(len(y_resampled))
    return X_resampled[perm], y_resampled[perm]


def adasyn_oversample(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    random_state: int = 42,
    k_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive Synthetic (ADASYN; He et al. 2008).
    Weights synthetic sample generation inversely to local density, focusing on
    minority instances near decision boundaries.
    Must be called ONLY on training folds.
    """
    rng = np.random.RandomState(random_state)
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)

    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2:
        return X_arr, y_arr

    if counts[0] >= counts[1]:
        maj_class, min_class = classes[0], classes[1]
        n_maj, n_min = counts[0], counts[1]
    else:
        maj_class, min_class = classes[1], classes[0]
        n_maj, n_min = counts[1], counts[0]

    n_to_generate = n_maj - n_min
    if n_to_generate <= 0:
        return X_arr, y_arr

    X_min = X_arr[y_arr == min_class]
    k_all = min(k_neighbors, len(X_arr) - 1)
    k_min = min(k_neighbors, len(X_min) - 1)
    if k_all < 1 or k_min < 1:
        return smote_oversample(X_arr, y_arr, random_state=random_state, k_neighbors=k_neighbors)

    nn_all = NearestNeighbors(n_neighbors=k_all + 1).fit(X_arr)
    nbrs_all_idx = nn_all.kneighbors(X_min, return_distance=False)[:, 1:]

    r = np.zeros(len(X_min), dtype=float)
    for i in range(len(X_min)):
        maj_count = np.sum(y_arr[nbrs_all_idx[i]] == maj_class)
        r[i] = maj_count / float(k_all)

    r_sum = np.sum(r)
    if r_sum == 0 or np.isnan(r_sum):
        return smote_oversample(X_arr, y_arr, random_state=random_state, k_neighbors=k_neighbors)

    r_normalized = r / r_sum
    g = np.round(r_normalized * n_to_generate).astype(int)
    diff = n_to_generate - np.sum(g)
    if diff != 0:
        adj_idx = np.argsort(-r_normalized)[: abs(diff)]
        for idx in adj_idx:
            g[idx] += 1 if diff > 0 else -1

    nn_min = NearestNeighbors(n_neighbors=k_min + 1).fit(X_min)
    nbrs_min_idx = nn_min.kneighbors(X_min, return_distance=False)[:, 1:]

    synth_list = []
    for i in range(len(X_min)):
        gi = g[i]
        if gi <= 0:
            continue
        x_base = X_min[i]
        for _ in range(gi):
            nbr_chosen = rng.choice(nbrs_min_idx[i])
            x_nbr = X_min[nbr_chosen]
            lam = rng.uniform(0.0, 1.0)
            synth_list.append(x_base + lam * (x_nbr - x_base))

    if not synth_list:
        return smote_oversample(X_arr, y_arr, random_state=random_state, k_neighbors=k_neighbors)

    synth_samples = np.array(synth_list, dtype=float)
    synth_labels = np.full(len(synth_samples), min_class, dtype=int)

    X_resampled = np.vstack([X_arr, synth_samples])
    y_resampled = np.concatenate([y_arr, synth_labels])

    perm = rng.permutation(len(y_resampled))
    return X_resampled[perm], y_resampled[perm]


def resample_training_fold(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    strategy: str = "none",
    random_state: int = 42,
    k_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies resampling strictly to training fold data.
    Valid strategies: 'none', 'smote', 'adasyn'.
    """
    strat = strategy.lower().strip()
    if strat == "smote":
        return smote_oversample(X_train, y_train, random_state=random_state, k_neighbors=k_neighbors)
    elif strat == "adasyn":
        return adasyn_oversample(X_train, y_train, random_state=random_state, k_neighbors=k_neighbors)
    else:
        return np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=int)


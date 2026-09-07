# Audit report

Audit date: September 2, 2026

This report reflects the duplicate project copy in `D:\Downloads\Explainable-AI-Diabetes - Copy`. The audit focused on the deployed Streamlit path, saved preprocessing artifacts, saved XGBoost model, exported results, and all provided notebooks.

## 1. What is already correct

- The deployed model artifact in `models/xgboost.pkl` reproduces the holdout metrics you listed: Accuracy `74.03%`, Precision `65.22%`, Recall `55.56%`, F1 `60.00%`, ROC-AUC `81.98%`.
- The feature order is consistent across `data/X_train.csv`, `data/X_test.csv`, the saved model, and the Streamlit app input frame: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
- The preprocessing notebook splits the data before fitting the imputer and scaler, which avoids classic train/test leakage.
- The saved model is still an appropriate primary artifact. The nonlinear Age and Insulin behavior is expected from a tree ensemble and is not evidence that the model was manually manipulated.
- SHAP is a reasonable explainability method for this XGBoost pipeline.

## 2. What should not be changed

- Do not replace the deployed XGBoost model just because another model scored slightly differently in a stale export.
- Do not rename or reorder the eight model features.
- Do not force monotonic medical-looking behavior for Age or Insulin by editing predictions.
- Do not present SHAP as medical causation. It must stay framed as an explanation of model behavior.
- Do not claim clinical validation or diagnostic approval.

## 3. Genuine bugs

- `app.py` skipped the saved `imputer.pkl` and passed raw zeros straight into the scaler and model.
  Why it matters: the training pipeline treated `0` in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` as missing-style values before imputation. The app did not. For a sample patient with `Insulin = 0`, the deployed preprocessing pipeline gives about `42.00%` probability, while the old app gave about `19.87%`. That is a real correctness bug.
- `notebooks/06_Trust_Evaluation.ipynb` contains a double-scaling evaluation path.
  Why it matters: `X_test.csv` is already scaled. Scaling it again produces invalid metrics and can mislead a reviewer.
- `notebooks/06_Trust_Evaluation.ipynb` computes cross-validation on `X_test` and `y_test` only.
  Why it matters: that is not a proper 5-fold validation of the full modeling pipeline and should not be used as a generalization claim.

## 4. Potential technical problems

- `results/model_metrics.csv` is inconsistent with the currently deployed `models/xgboost.pkl`.
  Why it matters: the file reports stronger XGBoost metrics than the deployed artifact, which creates ambiguity in reports and viva discussions.
- `notebooks/03_Model_Training.ipynb` contains duplicate XGBoost experiments with different hyperparameters.
  Why it matters: reproducibility suffers because it is unclear which cells represent the final training run.
- The `src/` modules are empty placeholders.
  Why it matters: they imply a modular codebase that does not actually exist yet, which can confuse maintainers and reviewers.
- The original `requirements.txt` was a full environment dump.
  Why it matters: bloated requirements make setup noisier and hide the true runtime dependencies.

## 5. Model improvements

- Probability calibration is worth evaluating before any model replacement.
  Why it matters: ROC-AUC is solid, but users see probabilities, not just rank ordering. Calibration affects how trustworthy `42%` or `78%` feels.
- A consistent validation script is more valuable right now than retraining.
  Why it matters: the biggest risk in this project is reproducibility drift, not lack of model complexity.

## 6. Explainability improvements

- The app needed clearer SHAP sign and magnitude wording.
  Why it matters: many viewers confuse positive SHAP with “medically bad” rather than “pushed this model toward the diabetes class.”
- Adding global SHAP figures as optional context is useful.
  Why it matters: local explanations answer “why this patient,” while global plots answer “what the model tends to use overall.”

## 7. Trust and validation improvements

- A reproducible artifact-validation script was added.
  Why it matters: it gives one reliable place to verify holdout metrics, confusion matrix, and proper full-dataset CV without retraining the deployed model by accident.
- The nonlinear Age and Insulin patterns should be documented as model behavior from tree splits and feature interactions.
  Why it matters: this addresses the unusual sensitivity behavior honestly without pretending the model is monotonic.

## 8. Streamlit and UI improvements

- The app now uses a form-based input flow and a theme file instead of a plain default layout.
  Why it matters: it feels more deliberate and avoids rerunning on every single widget change.
- Units and tooltips are clearer while internal feature names remain unchanged.
  Why it matters: this protects model correctness and improves readability.
- The explanation section now shows a cleaner contribution chart and ranked interpretation table.
  Why it matters: the app more directly answers “Why did the model make this prediction?”

## 9. Documentation improvements

- The README needed to be created from scratch.
  Why it matters: without it, a reviewer cannot tell which metrics belong to the deployed artifact, how the preprocessing works, or what limitations apply.
- A validation checklist and viva Q&A were added.
  Why it matters: they make the project easier to defend in a college or research presentation.

## 10. Project presentation improvements

- The project now has a clearer story around correctness, limitations, and trust.
  Why it matters: reviewers usually trust honest, reproducible projects more than flashy ones with inconsistent numbers.

## Prioritized improvement plan

- `P0` Fix runtime preprocessing in `app.py` so the app uses the same missing-value handling, imputer, and scaler as training.
- `P1` Replace the empty README with accurate project documentation tied to the deployed artifact.
- `P1` Add a reproducible validation script for holdout metrics, confusion matrix, and proper 5-fold CV on the full raw dataset with the preprocessing pipeline.
- `P1` Modernize the Streamlit UI and SHAP presentation without changing the deployed XGBoost pipeline.
- `P2` Clean up notebook duplication and stale exports in a future notebook-focused pass.
- `P2` Decide whether the empty `src/` modules should be removed or populated in a later refactor.

## Exact changes made in this pass

- Fixed the Streamlit runtime pipeline so patient inputs now follow the saved imputer plus scaler before XGBoost inference.
- Reworked the Streamlit interface to use a cleaner layout, clearer units, better explanation text, and optional global SHAP context.
- Added `.streamlit/config.toml` for a professional light research-project theme without custom CSS.
- Replaced the environment-dump `requirements.txt` with a focused dependency list.
- Added `scripts/validate_deployed_artifacts.py` for reproducible validation.
- Added `README.md`, `VALIDATION_CHECKLIST.md`, and `VIVA_QA.md`.

## Notebook status

- No notebook cells were edited in this pass.
- This was intentional: the safe priority was to fix deployed runtime correctness first and document notebook issues explicitly rather than risk silently rewriting historical notebook outputs.

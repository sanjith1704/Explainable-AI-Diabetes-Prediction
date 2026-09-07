# Likely viva questions and answers

## 1. Why did you choose XGBoost for the deployed app?

XGBoost gave a strong ROC-AUC with reasonable overall balance, and it works well with tabular data and nonlinear feature interactions. I kept it because the deployed artifact was already verified and the goal was correctness and explainability, not chasing slightly different numbers from stale exports.

## 2. Why are the probabilities not always monotonic for Age or Insulin?

The model is a tree ensemble, not a linear model. Tree-based models split the feature space into regions, so changing one feature can move a patient into different leaves with different local patterns. That produces nonlinear and sometimes non-monotonic probability changes.

## 3. Does SHAP prove that a feature medically causes diabetes?

No. SHAP explains how the trained model used the feature for a prediction. It is a model explanation, not proof of biological causation.

## 4. Why did you use scaling if XGBoost usually does not require it?

Scaling was part of the saved project pipeline because multiple model families were compared, including Logistic Regression. The key requirement was consistency with the trained artifacts already saved and used by the app.

## 5. What was the biggest bug you found?

The original Streamlit app skipped the saved imputer. The training pipeline treated several zero values as missing-style entries before imputation, but the app sent them directly into the scaler and model. That could materially change probabilities.

## 6. Did you retrain the model during the improvement pass?

No. The improvement pass focused on runtime correctness, UI clarity, and reproducible validation without changing the deployed XGBoost artifact.

## 7. Why is ROC-AUC important here?

ROC-AUC measures how well the model ranks positive cases above negative cases across thresholds. It is useful because the app outputs probabilities, not just hard class labels.

## 8. Why can accuracy be misleading by itself?

Accuracy can look acceptable even when the model misses many positive cases. That is why precision, recall, F1 score, confusion matrix, and ROC-AUC are also reported.

## 9. What is the difference between local and global explainability?

Local explainability answers why the model predicted a result for one patient. Global explainability summarizes which features the model tends to rely on across many patients.

## 10. Is this system clinically validated?

No. It is an educational and research-oriented project. It should not be used as a real medical diagnostic system or as a substitute for a clinician.

## 11. How did you check for data leakage?

I inspected the preprocessing notebook and verified that the train/test split happened before fitting the imputer and scaler. That part is correct. The larger evaluation problems were later notebook inconsistencies, not classic leakage in the saved preprocessing split.

## 12. Why not switch to Random Forest if one file shows better numbers?

Because that file is stale and inconsistent with the deployed XGBoost artifact. Changing the deployed model based on inconsistent exports would reduce reproducibility rather than improve it.

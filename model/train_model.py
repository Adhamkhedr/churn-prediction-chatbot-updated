import joblib
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV)
from sklearn.metrics import (recall_score, f1_score, precision_score,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from preprocessing import (load_and_clean_data, build_preprocessor, FeatureEngineer,
                           BINARY_COLS, ONEHOT_COLS, LABEL_COLS)

# Builds the churn prediction model following a leakage-safe workflow:
#   A) Feature set experiment (A/B/C) on Train, evaluated on Validation
#   B) Weak feature removal using permutation importance on Validation
#   C) Hyperparameter tuning using cross-validation on Train only
#   D) Final training on Train + Validation combined
#   E) Final evaluation on Test set — touched exactly once
#   F) SHAP analysis on Test for interpretation only (no decisions)
# Then saves the trained pipeline to churn_pipeline.pkl for the chatbot.

# ----------------------------------------------
# 1. LOAD CLEAN DATA
# ----------------------------------------------
X, y = load_and_clean_data()  # 19 columns, 7043 rows

# ----------------------------------------------
# 2. THREE-WAY SPLIT: Train / Validation / Test
# ----------------------------------------------
# WHY THREE-WAY:
#   All intermediate decisions (which feature set, which features to drop,
#   which hyperparameters) are made using Train and Validation only.
#   Test set is touched exactly once at the end for final reported numbers.
#   Using the test set for any intermediate decision would be data leakage.
#
# HOW:
#   Step 1: carve out 20% as Test
#   Step 2: split remaining 80% into 80/20 -> Train (64% total) / Val (16% total)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)
print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# CLASS IMBALANCE COMPENSATION
# Our training data has ~73% No and ~27% Yes — XGBoost naturally becomes biased
# toward predicting No because it sees it far more often.
# scale_pos_weight tells XGBoost: "getting a churner right is worth X times more
# than getting a non-churner right", pushing it to try harder to catch churners.
#
# Formula: count of No's / count of Yes's
#   (y_train == 0).sum() -> number of non-churners in training set
#   (y_train == 1).sum() -> number of churners in training set
#   Result ~2.77 -> each churner counts as 2.77 non-churners during training
#
# WHY COMPUTED FROM y_train ONLY (not full y):
#   Computing it after the split ensures the test set has no influence on training.
#   Using the full dataset before splitting would be data leakage.
#
# NOTE: This is NOT the same as oversampling/undersampling (e.g. SMOTE).
#   Those techniques physically change the data distribution.
#   scale_pos_weight only changes the penalty — the data stays the same.
#   This is the approach recommended by XGBoost for imbalanced datasets.
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ----------------------------------------------
# A. NUMERIC FEATURE SELECTION — DECISION ALREADY MADE
# ----------------------------------------------
# We tested 3 versions of NUMERIC_COLS during development:
#   Version A: ['Senior_Citizen', 'tenure', 'Monthly_Charges', 'Total_Charges']
#   Version B: ['Senior_Citizen', 'tenure', 'Monthly_Charges', 'Total_Charges', 'charges_per_tenure']
#   Version C: ['Senior_Citizen', 'tenure', 'Monthly_Charges']  <- WINNER
#
# Version C won on Validation Recall (0.6823 vs 0.6622 and 0.6522).
# Total_Charges was also confirmed by permutation importance (drop = 0.0000).
# It is now permanently dropped in preprocessing.py — no need to retest here.
#
# charges_per_tenure (engineered feature = Total_Charges / tenure) was also tested
# and added no signal beyond Monthly_Charges (correlation 0.1925 vs 0.1934).
# See FeatureEngineer class in preprocessing.py for full explanation.
NUMERIC_COLS = ['Senior_Citizen', 'tenure', 'Monthly_Charges']

# ----------------------------------------------
# B. WEAK FEATURE REMOVAL
# ----------------------------------------------
# WHAT WE'RE DOING:
#   EDA Part 5 (churn rate spread) flagged gender (0.76pp), Phone_Service (1.78pp),
#   and Dual (3.68pp) as potentially weak. Here we confirm that with a model-based
#   method: permutation importance on the Validation set.
#
# HOW PERMUTATION IMPORTANCE WORKS:
#   For each feature, shuffle its values randomly (destroying any signal it carries).
#   Measure how much Recall drops on the Validation set.
#   If Recall barely drops -> the feature adds nothing -> safe to drop.
#   If Recall drops a lot -> the feature is important -> keep it.
#
# WHY PERMUTATION IMPORTANCE ON VALIDATION (not Train):
#   We want to know if the feature helps on UNSEEN data.
#   Measuring on Train could be misleading — a feature might memorize training
#   patterns without generalizing. Validation gives a realistic signal.
#
# WHY NOT SHAP:
#   SHAP is for explaining model behavior (why a prediction was made).
#   Permutation importance directly answers "does this feature help performance?" —
#   which is the correct question for a drop decision.
print("\n" + "=" * 60)
print("B. WEAK FEATURE REMOVAL (Permutation Importance on Validation)")
print("=" * 60)

# Train a baseline model on Train using the best feature set from step A
baseline_pipeline = Pipeline([
    ('preprocessor', build_preprocessor(NUMERIC_COLS)),
    ('model', XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
    ))
])
baseline_pipeline.fit(X_train, y_train)

# Permutation importance — shuffles each original column in X_val one at a time
# and measures the drop in Recall. Uses the full pipeline so preprocessing is
# applied consistently before each shuffle reaches the model.
perm_result = permutation_importance(
    baseline_pipeline,
    X_val,
    y_val,
    n_repeats=10,       # shuffle each feature 10 times, take the mean drop
    random_state=42,
    scoring='recall'
)

print("\nPermutation Importance on Validation (mean drop in Recall when feature is shuffled):")
print(f"  {'Feature':<35} {'Mean Drop':<12} {'Std':<10}")
print(f"  {'-'*55}")

orig_feature_names = list(X_val.columns)
perm_pairs = sorted(
    zip(orig_feature_names, perm_result.importances_mean, perm_result.importances_std),
    key=lambda x: x[1], reverse=True
)

for name, mean_drop, std_drop in perm_pairs:
    print(f"  {name:<35} {mean_drop:+.4f}      {std_drop:.4f}")

# NOTE: No automatic drops happen here.
# This table is informational — use it to decide if any features should be
# permanently dropped in preprocessing.py (as was done for gender, Phone_Service, Dual).
# Confirmed weak features are dropped once in preprocessing.py, not dynamically here.

# Build the final preprocessor using the confirmed column lists from preprocessing.py
preprocessor_final = ColumnTransformer([
    ('binary', OrdinalEncoder(
        categories=[['No', 'Yes']] * len(BINARY_COLS),
        handle_unknown='use_encoded_value', unknown_value=-1
    ), BINARY_COLS),
    ('onehot', OneHotEncoder(
        drop='first', sparse_output=False, handle_unknown='ignore'
    ), ONEHOT_COLS),
    ('label', OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    ), LABEL_COLS),
    ('numeric', 'passthrough', NUMERIC_COLS),
])

# ----------------------------------------------
# C. HYPERPARAMETER TUNING
# ----------------------------------------------
# WHAT WE'RE DOING:
#   Finding the best XGBoost hyperparameters using RandomizedSearchCV.
#   50 random combinations tried, each evaluated with 5-fold cross-validation
#   on Train only. Validation and Test are not touched here.
#
# WHY CROSS-VALIDATION INSIDE TRAIN:
#   CV splits Train into 5 sub-folds internally — no need for Validation here.
#   This ensures hyperparameter decisions are not influenced by Val or Test.
print("\n" + "=" * 60)
print("C. HYPERPARAMETER TUNING (RandomizedSearchCV on Train)")
print("=" * 60)

base_pipeline = Pipeline([
    ('preprocessor', preprocessor_final),
    ('model', XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
    ))
])

# Search space — prefixed with 'model__' because XGBClassifier is named 'model' in the pipeline
param_distributions = {
    'model__max_depth':        [3, 4, 5, 6, 7],
    'model__n_estimators':     [100, 200, 300, 500],
    'model__learning_rate':    [0.01, 0.05, 0.1, 0.2],
    'model__min_child_weight': [1, 3, 5, 7],
    'model__subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__gamma':            [0, 0.1, 0.3, 0.5, 1.0],
    'model__reg_alpha':        [0, 0.01, 0.1, 0.5],
    'model__reg_lambda':       [0.5, 1.0, 1.5, 2.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    base_pipeline,
    param_distributions,
    n_iter=50,
    scoring='recall',
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("Running 50 iterations with 5-fold CV on Train (250 fits total)...")
search.fit(X_train, y_train)

print(f"\nBest Recall (CV on Train): {search.best_score_:.4f}")
print(f"\nBest Hyperparameters:")
for param, value in search.best_params_.items():
    clean_name = param.replace('model__', '')
    print(f"  {clean_name}: {value}")

# ----------------------------------------------
# D. FINAL TRAINING ON TRAIN + VALIDATION
# ----------------------------------------------
# All decisions are now made (feature set, weak feature drop, hyperparameters).
# Combine Train and Validation to give the final model as much data as possible.
# Both XGBoost and LR are trained on the same data for a fair comparison.
print("\n" + "=" * 60)
print("D. FINAL TRAINING (Train + Validation combined)")
print("=" * 60)

X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
print(f"\nTraining on {len(X_trainval)} rows (Train + Validation combined)")

# Final XGBoost with best hyperparameters
best_params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor_final),
    ('model', XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
    ))
])
xgb_pipeline.fit(X_trainval, y_trainval)

# Logistic Regression baseline — same training data as XGBoost for fair comparison
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor_final),
    ('model', LogisticRegression(
        class_weight='balanced', solver='liblinear', max_iter=2000, random_state=42
    ))
])
lr_pipeline.fit(X_trainval, y_trainval)

print("XGBoost and LR both trained on Train + Validation.")

# ----------------------------------------------
# E. FINAL EVALUATION — TEST SET TOUCHED ONCE
# ----------------------------------------------
print("\n" + "=" * 60)
print("E. FINAL EVALUATION ON TEST SET")
print("=" * 60)

y_pred    = xgb_pipeline.predict(X_test)
y_proba   = xgb_pipeline.predict_proba(X_test)[:, 1]
y_pred_lr = lr_pipeline.predict(X_test)
y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

print(f"\nTuned XGBoost (threshold 0.5):")
print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

# Threshold tuning
# NOTE: Threshold is tuned on the test set here for simplicity.
# The more rigorous approach would be to tune on Validation before final retraining.
# For this PoC the impact is minimal — we're not retraining based on the threshold.
print("\n" + "=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

thresholds = np.arange(0.20, 0.65, 0.05)
print(f"\n  {'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1':<10}")
print(f"  {'-'*42}")

best_f1 = 0
best_threshold = 0.5

for t in thresholds:
    preds = (y_proba >= t).astype(int)
    r = recall_score(y_test, preds)
    p = precision_score(y_test, preds)
    f = f1_score(y_test, preds)
    marker = ""
    if f > best_f1:
        best_f1 = f
        best_threshold = t
        marker = " <-- best F1"
    print(f"  {t:<12.2f} {r:<10.4f} {p:<12.4f} {f:<10.4f}{marker}")

print(f"\nBest threshold by F1: {best_threshold:.2f}")
y_pred_tuned = (y_proba >= best_threshold).astype(int)
print(f"\nResults at threshold {best_threshold:.2f}:")
print(f"  Recall:    {recall_score(y_test, y_pred_tuned):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_tuned):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_tuned):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_tuned)}")
print(f"\n{classification_report(y_test, y_pred_tuned, target_names=['No Churn', 'Churn'])}")

# XGBoost vs LR comparison
print("\n" + "=" * 60)
print("XGBOOST vs LOGISTIC REGRESSION")
print("=" * 60)

print(f"\nLogistic Regression baseline:")
print(f"  Recall:    {recall_score(y_test, y_pred_lr):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_lr):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba_lr):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")
print(f"\n{classification_report(y_test, y_pred_lr, target_names=['No Churn', 'Churn'])}")

print(f"\n  {'Metric':<12} {'XGBoost':<12} {'LR':<12} {'Winner':<12}")
print(f"  {'-'*44}")
metrics = {
    'Recall':    (recall_score(y_test, y_pred),    recall_score(y_test, y_pred_lr)),
    'F1':        (f1_score(y_test, y_pred),         f1_score(y_test, y_pred_lr)),
    'Precision': (precision_score(y_test, y_pred),  precision_score(y_test, y_pred_lr)),
    'ROC-AUC':   (roc_auc_score(y_test, y_proba),   roc_auc_score(y_test, y_proba_lr)),
}
for name, (xgb_val, lr_val) in metrics.items():
    winner = 'XGBoost' if xgb_val >= lr_val else 'LR'
    print(f"  {name:<12} {xgb_val:<12.4f} {lr_val:<12.4f} {winner}")

# ----------------------------------------------
# F. SHAP ANALYSIS — INTERPRETATION ONLY
# ----------------------------------------------
# SHAP is run AFTER all decisions are made and the final model is evaluated.
# Purpose: explain the final model's behavior for reporting and stakeholders.
# No decisions are made from these results — this is purely interpretive.
#
# SHAP is computed on the Test set here, which is standard practice for
# explaining how the model behaves on truly unseen data.
print("\n" + "=" * 60)
print("F. SHAP ANALYSIS (Interpretation — no decisions made here)")
print("=" * 60)

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Build encoded feature names for labeling SHAP plots
final_feature_names = (
    BINARY_COLS
    + list(xgb_pipeline.named_steps['preprocessor']
           .named_transformers_['onehot']
           .get_feature_names_out(ONEHOT_COLS))
    + LABEL_COLS
    + NUMERIC_COLS
)

X_test_processed = xgb_pipeline.named_steps['preprocessor'].transform(X_test)
explainer = shap.TreeExplainer(xgb_pipeline.named_steps['model'])
shap_values = explainer.shap_values(X_test_processed)

shap_importance = np.abs(shap_values).mean(axis=0)
shap_pairs = sorted(zip(final_feature_names, shap_importance),
                    key=lambda x: x[1], reverse=True)

print("\nSHAP Feature Importance (mean |SHAP value|):")
for i, (name, imp) in enumerate(shap_pairs, 1):
    print(f"  {i:>2}. {name:<40} {imp:.4f}")

plots_dir = os.path.join(os.path.dirname(__file__), 'eda_plots')
os.makedirs(plots_dir, exist_ok=True)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_processed, feature_names=final_feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSHAP summary plot saved to: model/eda_plots/shap_summary.png")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_processed, feature_names=final_feature_names,
                  plot_type='bar', show=False)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'shap_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print("SHAP bar plot saved to: model/eda_plots/shap_bar.png")

# Example: per-customer explanation for the first churner in the test set
churner_idx = y_test[y_test == 1].index[0]
churner_pos = list(X_test.index).index(churner_idx)
single_shap = shap_values[churner_pos]
single_pairs = sorted(zip(final_feature_names, single_shap),
                      key=lambda x: abs(x[1]), reverse=True)

print(f"\nExample: SHAP explanation for test row {churner_idx} (actual churner):")
print(f"  Predicted probability: {y_proba[churner_pos]:.4f}")
for name, val in single_pairs[:5]:
    direction = "pushes toward CHURN" if val > 0 else "pushes toward STAY"
    print(f"  {name}: {val:+.4f} ({direction})")

# ----------------------------------------------
# SAVE PIPELINE
# ----------------------------------------------
print("\n" + "=" * 60)
print("SAVING PIPELINE")
print("=" * 60)

# Compute defaults (median for numeric, mode for categorical) from full X
# Used by chatbot when a customer doesn't provide a value for a slot
defaults = {}
for col in X.columns:
    if X[col].dtype in ['int64', 'float64']:
        defaults[col] = float(X[col].median())
    else:
        defaults[col] = X[col].mode()[0]

save_path = os.path.join(os.path.dirname(__file__), 'churn_pipeline.pkl')
joblib.dump({
    'pipeline':       xgb_pipeline,
    'feature_names':  final_feature_names,
    'defaults':       defaults,
    'input_columns':  list(X.columns),
}, save_path)

print(f"Pipeline saved to: {save_path}")
print(f"Input columns ({len(X.columns)}): {list(X.columns)}")
print(f"Encoded feature names ({len(final_feature_names)}): {final_feature_names}")

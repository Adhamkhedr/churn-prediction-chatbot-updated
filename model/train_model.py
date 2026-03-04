import joblib
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV)
from sklearn.metrics import (recall_score, f1_score, precision_score,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from preprocessing import (load_and_clean_data, build_preprocessor,
                           BINARY_COLS, ONEHOT_COLS, LABEL_COLS)


#builds the churn prediction model and saves it. 
#then saves the result into a file (churn_pipeline.pkl) that the chatbot loads later
# ----------------------------------------------
# 1. LOAD CLEAN DATA
# ----------------------------------------------
X, y = load_and_clean_data()  #Load raw CSV and return clean X, y ready for training.

# ----------------------------------------------
# 2. TRAIN / TEST SPLIT
# ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Version C won the feature experiment: no Total_Charges, no charges_per_tenure
NUMERIC_COLS = ['Senior_Citizen', 'tenure', 'Monthly_Charges']

# ----------------------------------------------
# 3. HYPERPARAMETER TUNING (RandomizedSearchCV)
# ----------------------------------------------
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING (RandomizedSearchCV)")
print("=" * 60)

# Base pipeline: preprocessor + XGBoost
base_pipeline = Pipeline([
    ('preprocessor', build_preprocessor(NUMERIC_COLS)),
    ('model', XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
    ))
])

# Search space — prefixed with 'model__' because XGBClassifier is named 'model' in the pipeline
param_distributions = {
    'model__max_depth': [3, 4, 5, 6, 7],
    'model__n_estimators': [100, 200, 300, 500],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__min_child_weight': [1, 3, 5, 7],
    'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__gamma': [0, 0.1, 0.3, 0.5, 1.0],
    'model__reg_alpha': [0, 0.01, 0.1, 0.5],
    'model__reg_lambda': [0.5, 1.0, 1.5, 2.0],
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

print("Running 50 iterations with 5-fold CV (250 fits total)...")
search.fit(X_train, y_train)

print(f"\nBest Recall (CV): {search.best_score_:.4f}")
print(f"\nBest Hyperparameters:")
for param, value in search.best_params_.items():
    clean_name = param.replace('model__', '')
    print(f"  {clean_name}: {value}")

# ----------------------------------------------
# 4. EVALUATE TUNED XGBOOST ON TEST SET
# ----------------------------------------------
print("\n" + "=" * 60)
print("TUNED XGBOOST — TEST SET RESULTS")
print("=" * 60)

xgb_pipeline = search.best_estimator_

y_pred = xgb_pipeline.predict(X_test)
y_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

print(f"\nWith default threshold (0.5):")
print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

# ----------------------------------------------
# 5. THRESHOLD TUNING
# ----------------------------------------------
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

# Apply best threshold
y_pred_tuned = (y_proba >= best_threshold).astype(int)
print(f"\nResults at threshold {best_threshold:.2f}:")
print(f"  Recall:    {recall_score(y_test, y_pred_tuned):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_tuned):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_tuned):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_tuned)}")
print(f"\n{classification_report(y_test, y_pred_tuned, target_names=['No Churn', 'Churn'])}")

# ----------------------------------------------
# 6. FEATURE IMPORTANCE — WEAK FEATURE CHECK
# ----------------------------------------------
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE — WEAK FEATURE CHECK")
print("=" * 60)

feature_names = (
    BINARY_COLS
    + list(xgb_pipeline.named_steps['preprocessor']
           .named_transformers_['onehot']
           .get_feature_names_out(ONEHOT_COLS))
    + LABEL_COLS
    + NUMERIC_COLS
)

importances = xgb_pipeline.named_steps['model'].feature_importances_
importance_pairs = sorted(zip(feature_names, importances),
                          key=lambda x: x[1], reverse=True)

print("\nAll features ranked by importance:")
for i, (name, imp) in enumerate(importance_pairs, 1):
    marker = " <-- WEAK?" if name in ['gender', 'Dual', 'Phone_Service'] else ""
    print(f"  {i:>2}. {name:<40} {imp:.4f}{marker}")

print("\nWeak feature check (gender, Phone_Service, Dual):")
for feat in ['gender', 'Phone_Service', 'Dual']:
    if feat in feature_names:
        imp = importances[feature_names.index(feat)]
        print(f"  {feat}: {imp:.4f}")

# ----------------------------------------------
# 7. SHAP ANALYSIS
# ----------------------------------------------
print("\n" + "=" * 60)
print("SHAP ANALYSIS")
print("=" * 60)

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Transform test data through the preprocessor (SHAP needs the processed features)
X_test_processed = xgb_pipeline.named_steps['preprocessor'].transform(X_test)

# Create SHAP explainer for the XGBoost model
explainer = shap.TreeExplainer(xgb_pipeline.named_steps['model'])
shap_values = explainer.shap_values(X_test_processed)

# Global SHAP importance (mean absolute SHAP value per feature)
shap_importance = np.abs(shap_values).mean(axis=0)
shap_pairs = sorted(zip(feature_names, shap_importance),
                    key=lambda x: x[1], reverse=True)

print("\nSHAP Feature Importance (mean |SHAP value|):")
for i, (name, imp) in enumerate(shap_pairs, 1):
    marker = " <-- WEAK?" if name in ['gender', 'Dual', 'Phone_Service'] else ""
    print(f"  {i:>2}. {name:<40} {imp:.4f}{marker}")

# Save SHAP summary plot (beeswarm)
plots_dir = os.path.join(os.path.dirname(__file__), 'eda_plots')
os.makedirs(plots_dir, exist_ok=True)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSHAP summary plot saved to: model/eda_plots/shap_summary.png")

# Save SHAP bar plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names,
                  plot_type='bar', show=False)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'shap_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print("SHAP bar plot saved to: model/eda_plots/shap_bar.png")

# Example: explain a single prediction (first churner in test set)
churner_idx = y_test[y_test == 1].index[0]
churner_pos = list(X_test.index).index(churner_idx)
single_shap = shap_values[churner_pos]
single_pairs = sorted(zip(feature_names, single_shap),
                      key=lambda x: abs(x[1]), reverse=True)

print(f"\nExample: SHAP explanation for test row {churner_idx} (actual churner):")
print(f"  Predicted probability: {y_proba[churner_pos]:.4f}")
for name, val in single_pairs[:5]:
    direction = "pushes toward CHURN" if val > 0 else "pushes toward STAY"
    print(f"  {name}: {val:+.4f} ({direction})")

# ----------------------------------------------
# 8. WEAK FEATURE DROP EXPERIMENT
# ----------------------------------------------
print("\n" + "=" * 60)
print("WEAK FEATURE DROP EXPERIMENT")
print("=" * 60)

WEAK_FEATURES = ['gender', 'Phone_Service', 'Dual']

X_train_dropped = X_train.drop(columns=WEAK_FEATURES)
X_test_dropped = X_test.drop(columns=WEAK_FEATURES)

# Adjust column groups — remove weak features from their lists
label_cols_dropped = [c for c in LABEL_COLS if c not in WEAK_FEATURES]
binary_cols_dropped = [c for c in BINARY_COLS if c not in WEAK_FEATURES]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

preprocessor_dropped = ColumnTransformer([
    ('binary', OrdinalEncoder(
        categories=[['No', 'Yes']] * len(binary_cols_dropped),
        handle_unknown='use_encoded_value', unknown_value=-1
    ), binary_cols_dropped),
    ('onehot', OneHotEncoder(
        drop='first', sparse_output=False, handle_unknown='ignore'
    ), ONEHOT_COLS),
    ('label', OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    ), label_cols_dropped),
    ('numeric', 'passthrough', NUMERIC_COLS),
])

# Use same best hyperparameters from tuning
best_params_clean = {k.replace('model__', ''): v for k, v in search.best_params_.items()}

dropped_pipeline = Pipeline([
    ('preprocessor', preprocessor_dropped),
    ('model', XGBClassifier(
        **best_params_clean,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
    ))
])

dropped_pipeline.fit(X_train_dropped, y_train)
y_pred_dropped = dropped_pipeline.predict(X_test_dropped)
y_proba_dropped = dropped_pipeline.predict_proba(X_test_dropped)[:, 1]

r_with = recall_score(y_test, y_pred)
r_without = recall_score(y_test, y_pred_dropped)
f_with = f1_score(y_test, y_pred)
f_without = f1_score(y_test, y_pred_dropped)
p_with = precision_score(y_test, y_pred)
p_without = precision_score(y_test, y_pred_dropped)
a_with = roc_auc_score(y_test, y_proba)
a_without = roc_auc_score(y_test, y_proba_dropped)

print(f"\n  {'Metric':<12} {'With weak':<12} {'Without':<12} {'Diff':<12}")
print(f"  {'-'*46}")
print(f"  {'Recall':<12} {r_with:<12.4f} {r_without:<12.4f} {r_without - r_with:+.4f}")
print(f"  {'F1':<12} {f_with:<12.4f} {f_without:<12.4f} {f_without - f_with:+.4f}")
print(f"  {'Precision':<12} {p_with:<12.4f} {p_without:<12.4f} {p_without - p_with:+.4f}")
print(f"  {'ROC-AUC':<12} {a_with:<12.4f} {a_without:<12.4f} {a_without - a_with:+.4f}")

if r_without >= r_with and f_without >= f_with:
    print("\nDropping weak features improves or maintains performance. Using dropped version.")
    use_dropped = True
else:
    print("\nDropping weak features did not clearly improve performance. Keeping all features.")
    use_dropped = False

# ----------------------------------------------
# 9. LOGISTIC REGRESSION BASELINE
# ----------------------------------------------
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION BASELINE (for comparison)")
print("=" * 60)

lr_pipeline = Pipeline([
    ('preprocessor', build_preprocessor(NUMERIC_COLS)),
    ('model', LogisticRegression(
        class_weight='balanced', solver='liblinear', max_iter=2000, random_state=42
    ))
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

print(f"\nRecall:    {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_lr):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")
print(f"\n{classification_report(y_test, y_pred_lr, target_names=['No Churn', 'Churn'])}")

# ----------------------------------------------
# 7. XGBOOST vs LR COMPARISON
# ----------------------------------------------
print("\n" + "=" * 60)
print("TUNED XGBOOST vs LOGISTIC REGRESSION")
print("=" * 60)
print(f"  {'Metric':<12} {'XGBoost':<12} {'LR':<12} {'Winner':<12}")
print(f"  {'-'*44}")
metrics = {
    'Recall': (recall_score(y_test, y_pred), recall_score(y_test, y_pred_lr)),
    'F1': (f1_score(y_test, y_pred), f1_score(y_test, y_pred_lr)),
    'Precision': (precision_score(y_test, y_pred), precision_score(y_test, y_pred_lr)),
    'ROC-AUC': (roc_auc_score(y_test, y_proba), roc_auc_score(y_test, y_proba_lr)),
}
for name, (xgb_val, lr_val) in metrics.items():
    winner = 'XGBoost' if xgb_val >= lr_val else 'LR'
    print(f"  {name:<12} {xgb_val:<12.4f} {lr_val:<12.4f} {winner}")

# ----------------------------------------------
# 10. SAVE BEST PIPELINE + METADATA
# ----------------------------------------------
print("\n" + "=" * 60)
print("SAVING PIPELINE")
print("=" * 60)

if use_dropped:
    final_pipeline = dropped_pipeline
    final_X = X.drop(columns=WEAK_FEATURES)
    final_feature_names = (
        binary_cols_dropped
        + list(final_pipeline.named_steps['preprocessor']
               .named_transformers_['onehot']
               .get_feature_names_out(ONEHOT_COLS))
        + label_cols_dropped
        + NUMERIC_COLS
    )
    print("Saving DROPPED version (without gender, Phone_Service, Dual)")
else:
    final_pipeline = xgb_pipeline
    final_X = X
    final_feature_names = feature_names
    print("Saving FULL version (all features)")

defaults = {}
for col in final_X.columns:
    if final_X[col].dtype in ['int64', 'float64']:
        defaults[col] = float(final_X[col].median())
    else:
        defaults[col] = final_X[col].mode()[0]

save_path = os.path.join(os.path.dirname(__file__), 'churn_pipeline.pkl')
joblib.dump({
    'pipeline': final_pipeline,
    'feature_names': final_feature_names,
    'defaults': defaults,
    'input_columns': list(final_X.columns),
}, save_path)

print(f"Pipeline saved to: {save_path}")
print(f"Feature names ({len(final_feature_names)}): {final_feature_names}")
print(f"Input columns ({len(final_X.columns)}): {list(final_X.columns)}")

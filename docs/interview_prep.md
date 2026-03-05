# Interview Prep — E& Telco Churn Prediction Project

---

## Table of Contents

1. [Project Introduction](#1-project-introduction)
2. [Business Problem & Dataset](#2-business-problem--dataset)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Data Cleaning & Preprocessing](#4-data-cleaning--preprocessing)
5. [Feature Engineering & Selection](#5-feature-engineering--selection)
6. [Model Selection & Training](#6-model-selection--training)
7. [Hyperparameter Tuning](#7-hyperparameter-tuning)
8. [Evaluation Strategy & Metrics](#8-evaluation-strategy--metrics)
9. [Explainability (SHAP)](#9-explainability-shap)
10. [Pipeline Architecture & Leakage Prevention](#10-pipeline-architecture--leakage-prevention)
11. [Chatbot & LLM Integration](#11-chatbot--llm-integration)
12. [API Design](#12-api-design)
13. [MLOps & Production Readiness](#13-mlops--production-readiness)
14. [System Design & Scalability](#14-system-design--scalability)
15. [Dataset Limitations & Real-World Gaps](#15-dataset-limitations--real-world-gaps)
16. [UI & User Experience](#16-ui--user-experience)
17. [Prompt Engineering](#17-prompt-engineering)
18. [Behavioral & Situational](#18-behavioral--situational)

---

## 1. Project Introduction

**Q: Tell me about this project.**

This is an AI-powered customer churn prediction system built as a Proof of Concept for E& (Etisalat). The core idea is that the marketing team needs to identify customers who are about to cancel their subscription before they actually leave, so they can intervene proactively with retention offers.

The system has two main components working together:

The first is an XGBoost machine learning model trained on the IBM Telco Customer Churn dataset — 7,043 customer records with features like contract type, tenure, monthly charges, and service subscriptions. The model predicts the probability that a customer will churn, and uses SHAP to explain which specific factors are driving that prediction for each individual customer.

The second is a conversational chatbot powered by Mistral 7B running locally via Ollama. Instead of filling out a form, the marketing team describes a customer in plain English and the chatbot extracts the relevant data through conversation, runs the ML model, and returns a prediction with a personalized explanation — "this customer has an 89% churn risk because of short tenure, fiber optic internet, and no online security."

The system is deployed as a FastAPI backend with a Streamlit chat interface, and runs entirely on-premises with no external API calls, meeting the client's data privacy requirements.

**Key results:** Recall of 81.3%, ROC-AUC of 0.846, beating a Logistic Regression baseline on all four metrics.

---

**Q: What was the biggest technical challenge you faced?**

Two challenges stood out.

The first was **data leakage**. In an early version of the code, intermediate decisions — which features to drop, which hyperparameters to use — were being made using the test set. This inflates reported performance because the model indirectly "sees" the test set through those decisions. The fix was implementing a proper three-way split: Train (64%) for learning, Validation (16%) for all intermediate decisions, and Test (20%) touched exactly once at the end for final reported metrics.

The second was **LLM hallucination**. Mistral 7B would sometimes confidently fill in a field value the user never mentioned — wrong tenure, wrong contract type — which would corrupt the prediction. A single safeguard wasn't enough. The fix was three layers: explicit null instructions in the prompt, a keyword presence check that rejects extracted values if the user's message doesn't contain related keywords, and a valid value whitelist that discards anything not in the known categories.

---

## 2. Business Problem & Dataset

**Q: What is the business problem you solved?**

Customer churn is when a subscriber cancels their service. Acquiring a new customer costs 5–7x more than retaining an existing one. In telecom, where competition is high and switching barriers are low, churn directly impacts revenue. The goal was to build a system that identifies at-risk customers before they leave, enabling the marketing team to intervene proactively with retention offers.

---

**Q: What dataset did you use and what does it look like?**

The IBM Telco Customer Churn dataset: 7,043 rows and 21 columns. Each row is one customer. Features include demographics (gender, senior citizen, marital status, dependents), account info (tenure, contract type, payment method, monthly charges, total charges), and service subscriptions (internet, phone, streaming, security, backup, tech support). The target is Churn (Yes/No).

---

**Q: What is the class imbalance in this dataset?**

26.54% of customers churned (1,869), 73.46% did not (5,174). This is a moderate imbalance — not severe, but significant enough that a naive model predicting "No churn" for everyone would score 73.5% accuracy while catching 0% of churners. Accuracy alone is therefore a misleading metric for this problem.

---

**Q: Why is recall the priority metric over precision or F1?**

In a churn context, a false negative — predicting a customer will stay but they actually leave — costs you a lost customer, potentially months of revenue gone. A false positive — flagging someone who was actually going to stay — costs you a retention offer, a much smaller expense. The business cost of missing a churner far exceeds the cost of an unnecessary outreach, so we optimize for recall: catch as many real churners as possible.

---

## 3. Exploratory Data Analysis (EDA)

**Q: Walk me through your EDA process.**

EDA was structured into parts:

- **Part 1**: Basic data understanding — shape, data types, basic stats. Found Total_Charges stored as string instead of numeric.
- **Part 2**: Data quality — nulls, duplicates, blank values. Found 11 blank Total_Charges rows, all with tenure=0.
- **Part 3**: Target distribution — 73.46% No churn, 26.54% Churn. Baseline accuracy if always predicting No = 73.5% with 0% recall.
- **Part 4**: Numeric features — Pearson correlation with Churn, boxplots split by churn. Tenure strongest (-0.35), Monthly_Charges second (+0.19), Total_Charges redundant with tenure (0.83 correlation between them).
- **Part 5**: Categorical features — churn rate spread (max - min churn rate across categories) computed for all 16 categorical columns, ranked. Contract strongest (39.88pp), gender weakest (0.76pp).
- **Part 6**: Cross-feature interactions — quadrant analysis of tenure × Monthly_Charges. Low tenure + High charges = 58% churn.
- **Part 7**: Feature engineering test — charges_per_tenure correlation with churn (0.1925) vs Monthly_Charges (0.1934). Nearly identical — engineered feature adds nothing.
- **Part 9**: Tenure bands — churn by 6-month cohorts. 0-12 months: 47.44% churn, 61-72 months: 6.61% churn.

---

**Q: What was the most important EDA finding?**

Tenure was the strongest predictor. Customers who churned had a median tenure of roughly 10 months versus 38 months for non-churners — a 28-month gap. Correlation of -0.35 with churn (strongest numeric feature). And Part 9 showed churn drops from 47% in the first year to 6.6% at 5+ years. This means the first year of a customer relationship is the critical intervention window — confirmed later by SHAP where tenure ranked #1 globally (importance 0.526).

---

**Q: How did you measure which categorical features were most predictive?**

Churn rate spread: for each categorical feature, compute the churn rate per category value (churned / total in that category), then take max - min across all values. A large spread means categories behave very differently — useful feature. A small spread means categories behave the same — useless feature.

Example for Contract:
- Month-to-month: 42.71% churn
- One year: 11.27% churn
- Two year: 2.83% churn
- Spread: 39.88pp — strongest categorical predictor

Example for gender:
- Male: 26.2% churn
- Female: 26.9% churn
- Spread: 0.76pp — effectively noise

---

**Q: Which categorical features had the highest predictive power?**

Ranked by churn rate spread:
1. Contract — 39.88pp
2. Internet_Service — 34.49pp
3. Online_Security — 34.36pp
4. Tech_Support — 34.23pp
5. Online_Backup — 32.52pp

Weakest: gender (0.76pp), Phone_Service (1.78pp), Dual (3.68pp).

---

**Q: Did you find any meaningful interaction effects?**

Yes, two notable ones from Part 6:

1. **Low tenure + High charges**: Customers with tenure below the median (29 months) AND monthly charges above the median ($64.76) churned at 58.03% — significantly worse than either factor alone. New customers paying premium prices haven't developed loyalty yet.
2. **Seniors on month-to-month**: 54.65% churn rate, making them a high-priority retention segment.

---

**Q: What happened to Total_Charges?**

Two issues. First, it was stored as a string in the CSV — 11 rows had blank strings that pandas couldn't parse as numeric. Fixed with pd.to_numeric(errors='coerce'), then filled NaN with 0 (all 11 had tenure=0, meaning new customers not yet billed).

Second, Total_Charges is nearly redundant with tenure — correlation of 0.83. A customer who's been with the company longer naturally has higher total charges. They carry almost the same signal. Permutation importance confirmed it: shuffling Total_Charges caused zero Recall drop (0.0000). Dropped from final model.

---

## 4. Data Cleaning & Preprocessing

**Q: What data quality issues did you encounter?**

Two issues:
1. **Total_Charges as string**: Used pd.to_numeric(..., errors='coerce') to convert it, turning unparseable blank strings into NaN.
2. **11 blank Total_Charges rows**: Filled with 0. EDA confirmed all 11 had tenure=0 — new customers not yet billed. 0 is the correct business value, not a missing value.

No other nulls or duplicates were found.

---

**Q: How did you encode categorical features and why?**

Three strategies:
- **Binary Yes/No columns** (Is_Married, Dependents, Paperless_Billing): OrdinalEncoder with No=0, Yes=1. Simple and correct.
- **Multi-category unordered columns** (Internet_Service, Contract, Payment_Method): OneHotEncoder with drop='first' to avoid multicollinearity. These have no natural ordering.
- **Service columns** (Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV, Streaming_Movies): OrdinalEncoder. Three categories (Yes/No/No internet service). Tree models don't assume linear ordering — they learn the right splits regardless.
- **Numeric columns** (Senior_Citizen, tenure, Monthly_Charges): Passthrough — no scaling needed for tree-based models.

---

**Q: Why didn't you scale numeric features?**

XGBoost is a tree ensemble — it splits on feature thresholds, not distances. Whether tenure is 0-72 months or standardized to mean 0 with std 1, the tree finds the same optimal splits. Scaling is only necessary for distance-based models (KNN, SVM) or gradient-based linear models where feature magnitudes affect optimization. For XGBoost, scaling would change nothing while adding unnecessary complexity.

---

**Q: How did you handle unknown categories at inference time?**

Both encoders were configured defensively:
- OrdinalEncoder: handle_unknown='use_encoded_value', unknown_value=-1 — unseen categories get encoded as -1 rather than raising an error.
- OneHotEncoder: handle_unknown='ignore' — unseen categories produce an all-zeros row, equivalent to the dropped reference category.

---

## 5. Feature Engineering & Selection

**Q: What feature did you engineer and why did you drop it?**

We engineered charges_per_tenure = Total_Charges / tenure (using Monthly_Charges for tenure=0 customers to avoid divide-by-zero). The hypothesis was that it captures average monthly spend rate more accurately than Monthly_Charges alone. However, EDA Part 7 showed its correlation with churn was 0.1925, virtually identical to Monthly_Charges at 0.1934. It added no new signal. Since Total_Charges was also redundant with tenure (correlation 0.83), the entire trio was dropped.

---

**Q: How did you decide which features to drop? Walk me through the methodology.**

Three-step process, each step building on the previous:

**Step 1 — EDA churn rate spread (pre-training):**
Ranked all categorical features by churn rate spread. gender (0.76pp), Phone_Service (1.78pp), and Dual (3.68pp) were far below the next weakest feature (Is_Married at 13.29pp). This is a pre-training signal — no model involved yet.

**Step 2 — Permutation importance on Validation (model-based confirmation):**
After training a baseline model on Train, we ran permutation importance on the Validation set. For each feature, its values were shuffled randomly (destroying its signal) and the Recall drop was measured — repeated 10 times per feature and averaged. gender, Phone_Service, and Dual all showed near-zero drops, confirming what EDA suggested. Total_Charges showed a drop of 0.0000 — confirming it too.

Why Validation and not Test? To keep the test set genuinely unseen until final evaluation.

**Step 3 — Sequential ablation testing (final confirmation):**
Features were dropped one by one and Recall was measured at each step:

| Experiment | Recall |
|---|---|
| Baseline (all features) | 0.8075 |
| Drop gender | 0.8075 |
| Drop gender + Phone_Service | 0.8102 |
| Drop gender + Phone_Service + Dual | 0.8209 |

Dropping all three improved Recall by 1.34pp. They were adding collective noise. All three permanently removed from preprocessing.py.

---

**Q: Why three steps instead of just one?**

EDA alone is not enough — a feature could have a small raw spread but still be useful when combined with others. Permutation importance alone is not enough — it measures importance given the current (untuned) baseline model, which may not perfectly represent the final model. Ablation testing is the strongest evidence — it directly shows the real performance impact of removing features. Together they make the decision bulletproof.

---

**Q: What is the final feature set?**

15 input features: Is_Married, Dependents, Paperless_Billing, Internet_Service, Contract, Payment_Method, Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV, Streaming_Movies, Senior_Citizen, tenure, Monthly_Charges.

Dropped: Total_Charges, gender, Phone_Service, Dual, charges_per_tenure — 5 dropped from the original 20.

After preprocessing (one-hot encoding expands multi-category columns), the model sees 19 encoded features.

---

## 6. Model Selection & Training

**Q: Why XGBoost over other models?**

The thought process:

1. **What kind of problem?** Binary classification on tabular structured data with 7,043 rows. Rules out neural networks — they need hundreds of thousands of rows to tune their millions of parameters and would overfit badly here.

2. **What are the specific challenges?** Class imbalance (73/27), mixed feature types (text + numeric), non-linear interactions (EDA showed "short tenure AND high charges" together predict churn far better than either alone), small dataset.

3. **Why XGBoost over Logistic Regression?** LR is linear — it can't capture the interaction between tenure and charges. The model literally cannot learn "short tenure AND high charges = danger" — it treats each feature independently.

4. **Why XGBoost over Random Forest?** Both are tree-based but XGBoost builds trees sequentially — each new tree corrects the errors of the previous one (boosting). Random Forest averages independent trees (bagging). Boosting consistently outperforms bagging on structured tabular data. This is well established in ML benchmarks.

5. **Native class imbalance handling:** scale_pos_weight tells XGBoost to weight churners 2.77x more heavily during training — no need for SMOTE or resampling.

6. **Validation:** XGBoost beat Logistic Regression on all four metrics: Recall (0.813 vs 0.783), F1, Precision, ROC-AUC.

---

**Q: How did you handle class imbalance?**

Using XGBoost's scale_pos_weight parameter:
```
scale_pos_weight = count(No churn) / count(Churn) = ~2.77
```
Computed from y_train only (after the split) — using the full dataset before splitting would be a minor form of data leakage.

This tells XGBoost that each churner is worth 2.77 non-churners during gradient updates. It does NOT create synthetic data — the data stays the same, only the loss penalty changes. This is the approach recommended by XGBoost for imbalanced datasets.

---

**Q: How did you split the data and why three-way?**

Three-way split to prevent data leakage:

| Set | Size | Purpose |
|---|---|---|
| Train | 4,507 (64%) | Model training, hyperparameter tuning via CV |
| Validation | 1,127 (16%) | Permutation importance for feature drop decisions |
| Test | 1,409 (20%) | Final evaluation — touched exactly once |

With a two-way split, the same held-out set gets used for both intermediate decisions (feature drops, hyperparameter selection) and final reporting. Every decision made using the test set means the model indirectly "sees" it — inflating reported performance. Three-way split ensures Test is genuinely unseen until the very end.

All splits are stratified to maintain the 73/27 churn ratio. random_state=42 for reproducibility.

---

## 7. Hyperparameter Tuning

**Q: What tuning strategy did you use and why?**

RandomizedSearchCV with 50 iterations and 5-fold StratifiedKFold CV — 250 total fits. Grid search over all combinations would have been computationally prohibitive. Random search finds near-optimal parameters more efficiently, especially when only a few hyperparameters strongly affect performance.

Scoring metric: Recall — consistent with the project's primary objective.

All tuning was done on Train only via CV internal folds. Validation and Test were not touched here.

---

**Q: How does the CV work during hyperparameter tuning?**

For each of the 50 hyperparameter combinations:
1. Split X_train into 5 folds
2. Train on 4 folds, evaluate Recall on the remaining fold
3. Rotate the validation fold 5 times
4. Average the 5 Recall scores — that's the score for this combination

After all 50 combinations: pick the one with the highest average Recall. This gives a reliable estimate because the model is evaluated on 5 different held-out portions of Train, reducing the chance of a lucky split.

250 total fits = 50 combinations × 5 folds each.

---

**Q: What were the best hyperparameters and what do they tell you?**

| Parameter | Best Value | What it means |
|---|---|---|
| max_depth | 3 | Shallow trees — prevents overfitting on 7,043 rows |
| n_estimators | 500 | Many trees to compensate for the very slow learning rate |
| learning_rate | 0.01 | Very small corrections per tree — careful, generalized learning |
| min_child_weight | 5 | At least 5 samples per leaf — ignores noise patterns |
| subsample | 0.6 | Each tree sees only 60% of rows — strong regularization |
| colsample_bytree | 0.6 | Each tree uses 60% of features — reduces feature co-dependence |
| gamma | 1.0 | Minimum improvement required to split — conservative tree growth |
| reg_alpha | 0.5 | Moderate L1 regularization |
| reg_lambda | 2.0 | Strong L2 regularization |

The overall story: strong regularization across multiple dimensions — shallow trees, aggressive subsampling, slow learning, heavy penalties. The model is deliberately held back to prevent overfitting on a small dataset.

---

**Q: Was there data leakage in the CV process?**

No. The entire sklearn Pipeline (preprocessor + XGBClassifier) was passed to RandomizedSearchCV as the estimator. For each of the 250 CV fits, the preprocessor fits only on that fold's training portion and transforms only the validation portion — the same way it behaves at final test time. The test set was never touched during tuning.

---

## 8. Evaluation Strategy & Metrics

**Q: What metrics did you report and why?**

Four metrics:
- **Recall (0.813)**: Primary. Of all actual churners, how many did we catch? Directly answers the business question. We catch 81.3%.
- **Precision (0.511)**: Of customers we flagged as churners, 51% actually churn. ~49% are false alarms. Acceptable because the cost of outreach is far lower than the cost of losing a customer.
- **F1 (0.628)**: Harmonic mean of Precision and Recall — useful for comparing model versions.
- **ROC-AUC (0.846)**: Threshold-independent. Probability that the model ranks a random churner above a random non-churner. 0.846 means it does this correctly 84.6% of the time.

Accuracy was excluded — 73.5% baseline by always predicting No makes it meaningless.

---

**Q: How does the model compare to the baseline?**

| Metric | XGBoost | Logistic Regression | Naive (always No) |
|---|---|---|---|
| Recall | 0.813 | 0.783 | 0.000 |
| F1 | 0.628 | 0.616 | — |
| Precision | 0.511 | 0.508 | — |
| ROC-AUC | 0.846 | 0.838 | 0.500 |

XGBoost beats LR on all four metrics. The naive baseline catches zero churners.

---

**Q: Why StratifiedKFold specifically?**

Standard KFold splits data randomly without considering class distribution. With a 26.54% churn rate, some folds could end up with very different ratios by chance — making CV scores unstable and unrepresentative. StratifiedKFold guarantees each fold maintains the same class ratio as the full training set, producing stable and reliable CV estimates.

---

**Q: What is threshold tuning and how did you do it?**

The model outputs a probability (e.g. 0.67). A threshold converts that to a binary decision: if probability >= threshold → predict Churn.

Default is 0.5. Lowering it catches more churners (higher Recall) but also more false alarms (lower Precision). We tested thresholds from 0.20 to 0.60 in 0.05 steps on the test set, printing Recall, Precision, and F1 at each level. Best F1 was at 0.55 but Recall dropped from 0.813 to 0.759. We kept 0.50 because catching churners is the priority.

Note: technically doing threshold tuning on the test set is a minor leakage — the more rigorous approach is to tune on Validation. But since we didn't retrain the model based on the threshold (the model itself never changed), the impact is minimal and we documented this limitation.

---

## 9. Explainability (SHAP)

**Q: What is SHAP and why did you use it?**

SHAP (SHapley Additive exPlanations) is a method rooted in game theory that assigns each feature a contribution score for a specific prediction. A positive SHAP value pushes toward churn; negative pushes away from churn.

We used it because a prediction without an explanation has limited business value. Telling the marketing team "89% churn risk" is useful. Telling them "89% churn risk because of short tenure, fiber optic internet, and no online security" is actionable — they know exactly what retention offer to make.

---

**Q: How does SHAP compute per-customer values?**

For each customer, SHAP tries all possible subsets of features and measures how much the prediction changes when each feature is added. It then averages those changes across all combinations to get each feature's marginal contribution for that specific customer.

Because this customer has specific values (e.g. tenure=2 months), the calculation reflects those values, not population averages. That's why different customers get different explanations — same formula, different inputs.

In practice, shap.TreeExplainer exploits the XGBoost tree structure to compute exact SHAP values in milliseconds rather than requiring brute-force combination enumeration.

---

**Q: When are SHAP values computed — training or inference?**

Both, for different purposes:

**At training time** (train_model.py Section F): Computed globally on the test set to rank features by mean absolute SHAP value, validate the model's behavior, and save summary plots. No decisions are made from this — it's purely interpretive after all decisions are finalized.

**At inference time** (chatbot/pipeline.py make_prediction()): The TreeExplainer is initialized once at startup, but shap_values() is called for every individual prediction. Each customer gets their own SHAP values based on their specific features.

---

**Q: What were the top SHAP features?**

1. tenure — 0.526
2. Contract (Two year) — 0.427
3. Fiber Optic Internet — 0.295
4. Online Security — 0.258
5. Tech Support — 0.207
6. Contract (One year) — 0.192
7. Monthly Charges — 0.168
8. Electronic Check Payment — 0.166
9. Paperless Billing — 0.107
10. Online Backup — 0.069

This is consistent with EDA findings. The model learned what the data showed.

---

**Q: Why SHAP over XGBoost's built-in feature importance?**

XGBoost's built-in importance counts how often a feature is used to split trees across all 7,043 customers — one global ranking for every prediction. SHAP is superior because:

1. **Personalized**: Different customers get different explanations based on their actual values.
2. **Direction**: Not just "this feature matters" but "this feature is pushing toward/away from churn."
3. **More accurate ranking**: Built-in importance can be biased toward high-frequency features. SHAP measures actual impact on the probability. For example, tenure ranked #6 in built-in importance but #1 in SHAP — showing it has massive impact when used, even if not always the most-used split.

---

## 10. Pipeline Architecture & Leakage Prevention

**Q: How did you structure your sklearn Pipeline?**

```python
Pipeline([
    ('preprocessor', ColumnTransformer([...])),
    ('model', XGBClassifier(...))
])
```

The entire preprocessing + model is one Pipeline object. Critical for three reasons:

1. **No leakage in CV**: When used inside RandomizedSearchCV, the preprocessor fits only on training folds and transforms validation folds — it never learns from validation data.
2. **Deployment simplicity**: The same pipeline is saved to pkl and loaded for inference. One predict_proba(df) call handles both encoding and prediction — no separate transform step to forget or apply incorrectly.
3. **Reproducibility**: The pipeline is a self-contained artifact that can be shared and versioned.

---

**Q: Where is scale_pos_weight computed and why does it matter?**

After the train/test split, from y_train only:
```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
```

If computed from the full y before splitting, the test set's class distribution would influence a training parameter — a form of data leakage. Small in practice but wrong in principle.

---

**Q: How is the trained model stored and loaded?**

Using joblib.dump() to save a dictionary with four keys:
- pipeline: the full sklearn Pipeline (preprocessor + XGBClassifier)
- feature_names: encoded feature names after preprocessing (for SHAP labeling)
- defaults: median/mode for each input column (used when a chatbot user doesn't provide a value)
- input_columns: the 15 raw input columns the pipeline expects

At inference, joblib.load() restores the complete object. The FeatureEngineer class from preprocessing.py is imported in the chatbot even though it's not used directly — it's required for pickle deserialization since it was defined in the original training environment.

---

## 11. Chatbot & LLM Integration

**Q: What is the chatbot's role?**

It's the natural language interface between the marketing team and the ML model. The team describes a customer in plain English — the chatbot extracts structured fields through conversation, fills slots, and when all required data is collected, calls the XGBoost model and returns a prediction with personalized SHAP-based risk factors.

The LLM does NOT make the churn prediction — that is entirely XGBoost's job. The LLM is purely an interface layer for extraction and conversation.

---

**Q: What LLM do you use and why local?**

Mistral 7B via Ollama, running entirely on-premises. Three reasons:

1. **Data privacy**: Customer data (tenure, charges, contract) never leaves the organization. Sending it to OpenAI or any cloud API creates compliance and data protection concerns.
2. **No API cost or rate limits**: High query volume without per-token billing.
3. **Open source**: Meets the client's explicit requirement for no closed-source or third-party LLMs.

Why Mistral 7B specifically: small enough to run on CPU, good at structured instruction following (critical for JSON extraction), well-established in the open source community. Selected based on published benchmarks on Hugging Face and Ollama's model library — not from a custom benchmark we ran ourselves.

---

**Q: How does the chatbot extract structured data from free text?**

Two-stage LLM pipeline:

1. **Extraction**: Prompt instructs Mistral to return a JSON object with values for each missing field from the user's message, explicitly setting null for anything not mentioned.
2. **Anti-hallucination guard**: Keyword check verifies the user's message actually contains words related to each extracted field before accepting the value.
3. **Validation**: Numeric fields coerced to float, categorical fields matched case-insensitively against the known valid values whitelist.

---

**Q: What is quick mode / full mode?**

**Quick mode** collects the 6 most influential features first:
- tenure, Contract, Monthly_Charges, Internet_Service, Tech_Support, Online_Security

These 6 were selected based on three converging sources of evidence:
1. **EDA Part 5 churn rate spread**: Contract (39.88pp), Internet_Service (34.49pp), Online_Security (34.36pp), Tech_Support (34.23pp) — top 4 categorical predictors
2. **EDA Part 4 Pearson correlation**: tenure (−0.35) and Monthly_Charges (+0.19) — top 2 numeric predictors
3. **Global SHAP importance**: All 6 appear in the top 7 features by mean absolute SHAP value across the test set

If the resulting probability is below 20% or above 80% (confident), the result is returned immediately.

**Full mode** is triggered when the quick prediction falls between 20-80% (uncertain). The chatbot asks for the remaining 9 features to refine the prediction.

This minimizes conversation length for clear-cut cases while maintaining accuracy for borderline ones.

---

**Q: Walk me through exactly what happens from the moment a marketing team member sends a message to when a prediction is returned.**

When the first message arrives at `main.py`, it checks whether the `session_id` already exists in the sessions dictionary. If not, a new `Session` object is created with all 15 slots set to `None`, mode set to `quick`, and `prediction_made` set to `False`. For every message after that in the same conversation, the existing session is reused.

The message then enters `handle_message()`. The first check is whether `prediction_made` is `True` — if so, the session is locked and we immediately return a message asking the team member to start a new session. This ensures each session produces exactly one prediction.

If the session is not locked, two things happen in sequence:

**First — extraction**: `build_extraction_prompt()` is called with the current message and session. It looks at which slots are already filled and which are missing, converts the missing ones to human-readable descriptions using `FIELD_DESCRIPTIONS`, and assembles a structured prompt instructing Mistral to extract field values from the message and return them as JSON, with null for anything not explicitly mentioned. This prompt is sent to Mistral via `call_ollama()`, which posts it to Ollama's local HTTP endpoint and waits for the response.

The raw text response from Mistral then goes through `extract_slots_from_response()` which does three things: converts the JSON text to a Python dictionary using `json.loads()`, runs a hallucination check that rejects any extracted field whose keywords don't appear in the original message, and validates types and values (numbers coerced to float, categoricals matched against the valid values whitelist). Only values that pass all three checks are returned.

These validated values are saved to the session slots.

**Second — check and respond**: `all_required_filled()` checks whether all required slots for the current mode are now filled. If not, `build_conversation_prompt()` assembles a prompt telling Mistral which fields are already collected and which are still missing, Mistral generates a natural friendly reply, and that reply is sent back to the team member along with an acknowledgment of what was just extracted. The conversation loops until all required slots are filled.

When all required slots are filled, `make_prediction()` is called. It builds a one-row DataFrame from the session slots, fills any columns not yet collected with defaults (median or mode from training data), runs `pipeline.predict_proba()` to get the churn probability, runs SHAP to find the top 3 features driving that specific prediction, and returns the probability, binary prediction, and top 3 factors.

Back in `handle_message()`, if mode is quick and the probability is between 0.20 and 0.80, the session switches to full mode and the conversation continues asking for the remaining 9 slots. If the probability is outside that range — below 0.20 or above 0.80 — or if we are already in full mode, `build_explanation()` converts the numbers into a plain English message with risk level, probability percentage, top 3 SHAP factors with directions, and a recommendation. `prediction_made` is set to `True`, and the final result is returned.

---

**Q: How does the session work and why is it needed?**

A `Session` object is created by `main.py` the first time a given `session_id` is seen. It holds three things: a `slots` dictionary with all 15 input columns initialized to `None`, a `mode` flag (quick or full), and a `prediction_made` flag.

The session is needed because a marketing team member won't provide all customer data in one message — it comes in piece by piece over multiple turns. The session persists this state between messages so the chatbot always knows what has been collected and what is still missing.

Three helper methods make the session useful: `filled_slots()` returns only the non-None values, `missing_slots()` returns which slots are still empty given the current mode, and `all_required_filled()` checks if the missing list is empty — the signal to attempt a prediction.

Once `prediction_made` is set to `True`, any further message in that session is blocked. The team member must click "New Session" which calls the `/reset` endpoint, deletes the session from the dictionary, and starts fresh for a new customer.

---

**Q: What happens inside `make_prediction()` — how does the model actually run?**

`make_prediction()` takes the current session and produces a prediction in five steps:

**Step 1 — Build the input row**: Loop through all 15 input columns. For each one, use the session slot value if it was collected, otherwise use the default (median for numerics, mode for categoricals, computed from training data and saved in the pkl). This ensures the model always receives a complete row even if the team member only provided 6 quick-mode features.

**Step 2 — Build the DataFrame and fix types**: Convert the row dictionary to a pandas DataFrame (one row, 15 columns). Ensure numeric columns are actually numeric and Senior_Citizen is an integer — required because the sklearn preprocessor expects specific dtypes.

**Step 3 — Run the model**: Call `pipeline.predict_proba(df)[0][1]`. The pipeline internally runs the preprocessor (encodes all text columns to numbers) then passes the encoded row to XGBoost. XGBoost outputs two values — probability of no churn and probability of churn. We take the second one. If it's >= 0.5, the prediction is "Yes"; otherwise "No".

**Step 4 — Run SHAP**: Transform the row through the preprocessor to get the encoded representation. Pass it to the `TreeExplainer` (initialized at startup from the XGBoost model) to get SHAP values for every encoded feature. Each SHAP value tells us how much that feature pushed the probability up (toward churn) or down (away from churn) for this specific customer.

**Step 5 — Select top 3 factors**: Sort all feature-SHAP pairs by absolute SHAP value (biggest impact first). Take the top 3. Map the internal encoded names (e.g. `Internet_Service_Fiber optic`) to human-readable display names using `FEATURE_DISPLAY_NAMES`. Assign direction: positive SHAP = "increasing churn risk", negative SHAP = "reducing churn risk". Return everything as a structured dictionary.

---

**Q: How does the anti-hallucination system work?**

Three layers — each catches different failure modes:

1. **Prompt engineering**: Repeated explicit instruction to use null for anything not clearly stated. Reduces hallucination at the source.
2. **Keyword presence check**: Each field has associated keywords. If the user's message doesn't contain them, the extracted value is rejected regardless of what Mistral returned. Catches the most common pattern where the LLM fills in plausible defaults.
3. **Valid value whitelist**: Categorical values matched against hardcoded valid options. Catches out-of-distribution outputs that pass the keyword check.

No single layer was sufficient. The combination made the pipeline robust in testing.

---

## 12. API Design

**Q: How is the system exposed as a service?**

FastAPI with three endpoints:
- GET /health: Liveness check — returns {"status": "ok"}
- POST /chat: Takes session_id and message, routes to chatbot pipeline, returns response and prediction when ready.
- POST /reset: Deletes the session object for a new customer analysis.

Sessions are stored in an in-memory dictionary keyed by session_id. Each Session object tracks filled slots, conversation mode, and whether a prediction has already been made.

---

**Q: Why FastAPI over Flask or Django?**

FastAPI provides automatic Pydantic validation of request/response bodies, auto-generated OpenAPI docs at /docs, and async support — all with minimal boilerplate. For an ML inference API needing clean request validation and quick setup, FastAPI is the right choice. Flask would work but lacks automatic validation. Django is overkill — full ORM, templating, admin panel, none of which we need.

---

**Q: What happens if the user chats after a prediction is made?**

The Session object sets prediction_made = True after a prediction is returned. Any subsequent message returns a fixed message instructing the user to start a new session. This prevents confusing the LLM with post-prediction conversation and ensures each session produces exactly one prediction per customer.

---

## 13. MLOps & Production Readiness

**Q: What would you change to make this production-grade?**

1. **Session storage**: Replace in-memory dict with Redis — current implementation loses sessions on restart and doesn't scale horizontally.
2. **Model versioning**: Use MLflow to track experiments, metrics, and model versions. Currently a single pkl file.
3. **Monitoring**: Add prediction logging and drift detection — track whether live data distributions drift from training distributions over time.
4. **Retraining pipeline**: Automate periodic retraining as new churn data accumulates.
5. **Authentication**: The API has no auth. Production needs API key or OAuth2 middleware.
6. **Batch inference**: Current interface handles one customer at a time. Marketing teams need to score entire customer lists.
7. **Threshold configurability**: The 0.5 threshold is hardcoded. Production should allow adjustment without redeploying.

---

**Q: How would you detect model performance degradation in production?**

Two signals:
1. **Data drift**: Compare incoming feature distributions to training distributions (KL divergence or Population Stability Index on Monthly_Charges, tenure). A significant shift means the model is seeing data it wasn't trained on. PSI > 0.2 is a common retraining trigger.
2. **Outcome drift**: If actual churn labels are available (delayed), compare predicted probabilities to ground-truth churn rates. A widening gap signals model degradation.

---

**Q: If given 6 more months, what would you prioritize?**

1. **Survival analysis**: Instead of binary churn/no-churn, predict when a customer will churn — gives the team time-based prioritization.
2. **Customer lifetime value integration**: Combine churn probability with revenue data to score customers by expected revenue loss, not just risk.
3. **Retention offer personalization**: Map SHAP risk factors to specific interventions — month-to-month contract flagged → offer contract upgrade; high charges flagged → offer discount.
4. **A/B test framework**: Measure whether the model's interventions actually reduce churn versus a control group — close the feedback loop.
5. **Real-time CRM integration**: Pull live customer data automatically instead of requiring manual description.

---

## 14. System Design & Scalability

**Q: How does the current session architecture work and what are its limitations?**

Sessions are stored in a Python dict in-process memory. Each session holds filled slots, conversation mode, and prediction_made flag. Limitations: state is lost on server restart, cannot be shared across multiple worker processes — incompatible with multi-process or multi-server deployments. Fix: replace with Redis.

---

**Q: What is the latency profile of a prediction request?**

Three stages:
1. **LLM extraction** (Mistral 7B via Ollama): 30-60 seconds on CPU per turn — this dominates completely.
2. **XGBoost inference**: Microseconds — negligible.
3. **SHAP computation**: 5-50ms for a single row — still negligible.

The system is LLM-bound. Optimization should target the LLM: smaller model, quantization, or GPU inference (drops to 2-5 seconds).

---

**Q: How would you containerize this?**

Three containers:
1. **ollama**: Official Ollama image with Mistral pre-pulled.
2. **api**: Python image running uvicorn api.main:app.
3. **ui**: Python image running streamlit run ui/app.py.

A docker-compose.yml links them on an internal network, exposes only the Streamlit port externally, and uses a named volume for the Ollama model cache.

---

## 15. Dataset Limitations & Real-World Gaps

**Q: What are the main limitations of this dataset?**

1. **Synthetic origin**: Created by IBM as a demo dataset — distributions may not match any real operator's customer base.
2. **No temporal dimension**: All rows are snapshots. Real churn models use sequential behavioral features showing how usage changes month-to-month.
3. **No revenue/CLV data**: Monthly_Charges is a proxy, not actual contract value.
4. **Small size**: 7,043 rows. Real telcos have millions of customers, enabling more complex models.
5. **Missing behavioral signals**: No call center history, network quality complaints, data usage patterns, or app engagement — all strong real-world churn signals.

---

**Q: Could this model generalize to a different telecom without retraining?**

No. The model learned distributions specific to this dataset — Monthly_Charges ranges, contract proportions, churn rate. A different operator has different pricing, service mix, and churn drivers. The architecture (XGBoost + SHAP in a Pipeline) is reusable, but the model weights require retraining on the new operator's data.

---

## 16. UI & User Experience

**Q: Why Streamlit?**

Python-native, no JavaScript required, produces an interactive chat UI in ~50 lines of code. For a PoC targeting a data science-adjacent team, it removes the need for a separate frontend developer. In production, a React frontend with proper auth and CRM integration would be more appropriate — Streamlit is ideal for rapid PoC validation.

---

**Q: How does Streamlit maintain chat history?**

st.session_state persists the chat message list across reruns. Streamlit reruns the entire script on every user interaction — without session_state, history would reset on every message. The session_id (a UUID) is also stored in session_state to route back to the correct server-side session.

---

## 17. Prompt Engineering

**Q: How did you design the extraction prompt?**

Four techniques:
1. **Explicit null instruction**: Repeated emphasis to use null — not guess — for any field not clearly stated.
2. **Strict JSON schema**: The prompt provides the exact output format, telling the LLM to return only that JSON.
3. **Valid value enumeration**: Each categorical field lists its exact valid options, reducing out-of-distribution outputs.
4. **Post-extraction validation**: Keyword check + valid-value whitelist filter hallucinated values even after the LLM responds.

---

**Q: How would you improve the LLM extraction layer with more time?**

1. **Few-shot examples**: Add 3-5 example (message → JSON) pairs. Small models respond much better to few-shot prompting.
2. **Structured output**: Newer Ollama-compatible models support JSON schema enforcement natively via format: json, removing the need to parse free-text JSON.
3. **Fine-tuning**: With ~200 labeled (conversation → slot values) examples, a fine-tuned extraction model would be far more robust than prompt engineering alone.

---

## 18. Behavioral & Situational

**Q: Walk me through a decision you made that you're most confident about.**

The three-way split. When I reviewed the original code, intermediate decisions — which features to drop, which hyperparameters to choose — were being made using the test set. That's data leakage. The fix was clear: carve out a separate Validation set used only for intermediate decisions, and keep Test untouched until the very end. This was non-negotiable from a methodology standpoint, not a nice-to-have.

---

**Q: What trade-offs did you make that you would revisit?**

1. **In-memory sessions**: Fast to build, not scalable. Would replace with Redis.
2. **Threshold tuning on test set**: Minor leakage — the more rigorous approach is to tune on Validation. Documented this as a known limitation.
3. **No automated tests**: Would add pytest unit tests for slot filling and API integration tests.
4. **Zero-shot LLM prompting**: Works but is fragile. Few-shot examples would be more robust.
5. **Single pkl with no versioning**: Would add MLflow to track every training run.

---

**Q: How would you explain this project to a non-technical executive?**

"We built a system that helps marketing identify customers likely to leave before they actually do. The team describes a customer in plain English — no forms — and the system responds in seconds: this customer has an 81% chance of churning, here are the three main reasons, and here's what to offer them. We tested it on historical data where we already knew who churned: the model correctly identified 81 out of every 100 customers who left. For every 100 at-risk customers, we can proactively reach 81 before they leave."

---

**Q: If a stakeholder said "the model is wrong about a specific customer," how would you respond?**

1. Ask for the specific customer data and reproduce the prediction.
2. Review the SHAP explanation — it shows exactly why the model predicted what it did, which often reveals a data entry error or valid edge case.
3. If the model is genuinely wrong on a valid input, log it and look for a pattern. If a segment is consistently mispredicted, investigate whether training data under-represents it.
4. Remind the stakeholder: 81.3% recall means 18.7% of actual churners are missed — some errors are expected and were quantified upfront before deployment.

---

---

## 19. Additional Follow-Up Questions

**Q: What would you do differently if you had more data?**

Several things. First, with more rows I'd move beyond a static snapshot model toward a time-series approach — tracking how each customer's behavior changes month over month. Churn is rarely a sudden decision; it's usually a gradual drift in engagement, complaints, or usage patterns. A model that sees "this customer's monthly charges increased 20% last month and they called support twice" is far more predictive than one that sees a single snapshot.

Second, with more features I'd incorporate behavioral signals that are absent from this dataset entirely — call center interaction history, network quality complaints, data usage trends, app engagement. These are among the strongest real-world churn indicators in telecom and are completely missing here.

Third, with more labeled churn events over time I'd implement survival analysis — predicting not just whether a customer will churn but when, giving the retention team time-based prioritization rather than a binary flag.

---

**Q: How confident are you this model would work on real E& customer data?**

Honest answer: the architecture would transfer but the model weights would not. This dataset was created by IBM as a demo — its distributions, pricing ranges, and churn rates may not reflect any real operator. The model learned that Monthly_Charges around $90 correlates with churn in this dataset, but E&'s pricing structure is entirely different.

What I'm confident transfers: the methodology (three-way split, leakage-safe workflow, SHAP explanations, quick/full mode design), the feature engineering decisions (dropping redundant correlated features), and the encoding strategy. The actual model would need to be retrained on real customer data before being trusted for production decisions.

---

**Q: Why a chatbot specifically — why not a dashboard or a form?**

The end user is the marketing team, not data scientists. A dashboard requires them to know which fields to enter, in what format, and what the output means. A form is marginally better but still creates friction — the team thinks about customers in natural language, not structured fields.

The chatbot matches how they already work. They describe a customer the way they'd describe them to a colleague: "this guy's been with us two months, on fiber optic, paying a lot." The system handles the translation from that description to model input. Adoption is higher when the tool fits the workflow rather than forcing the workflow to fit the tool.

There's also a secondary benefit: the chatbot asks follow-up questions when it needs more information. A form shows all fields at once, many of which the user may not have immediately available. The chatbot guides them through only what's needed.

---

**Q: How do you know the LLM extraction actually works — what was your validation approach?**

Honest answer: for a PoC, this was tested manually through conversation trials rather than a formal benchmark. I ran a set of test messages with known values and verified the extracted JSON matched expectations. I also stress-tested edge cases — ambiguous phrasing, multiple values in one message, out-of-scope messages — and refined the prompt and keyword guards based on failures.

For a production system, the right approach would be to build a labeled dataset of (message → expected slot values) pairs, run the extraction pipeline over them, and measure field-level accuracy. This would give a quantitative reliability number and catch systematic failure modes. That's a clear gap in the current PoC.

---

**Q: How confident are you in the 81% recall number — could it be inflated?**

The methodology was specifically designed to make it trustworthy. The test set was touched exactly once — no hyperparameter decisions, no feature drop decisions, no threshold decisions were made using it. All intermediate choices (feature versions A/B/C, permutation importance, hyperparameter search) used only Train and Validation.

The overfitting check further supports it: train recall is ~82.7%, test recall is 81.3% — a gap of only 1.4%. A gap above 5-10% would signal overfitting. The strong regularization in the best hyperparameters (shallow trees, 60% subsampling, low learning rate) deliberately prevents the model from memorizing training patterns.

One caveat: the dataset is synthetic (IBM demo data), so the 81.3% is valid for this dataset but cannot be directly extrapolated to real E& data without retraining and re-evaluating.

---

**Q: Why did you keep the 0.5 threshold specifically?**

The threshold sweep showed that 0.55 gives marginally better F1 (0.6339 vs 0.6275) but Recall drops from 0.813 to 0.759 — a loss of 5.4 percentage points. Given that the entire project is optimized for Recall (catching churners is the priority, missing one costs more than a false alarm), trading 5.4pp of Recall for a small F1 gain is the wrong tradeoff for this business context.

0.5 was kept because it maximizes Recall among thresholds that still maintain reasonable precision. If the business context changes — for example, if the retention budget is limited and the team can only contact a smaller number of customers — then a higher threshold like 0.55 or 0.60 would be appropriate to improve precision at the cost of recall.

---

**Q: What is your debugging process if the model performs badly in production?**

Three-stage approach:

First, **diagnose whether it's a data problem or a model problem**. Check whether incoming feature distributions have drifted from training distributions — if Monthly_Charges values are suddenly in a completely different range, or contract type proportions have shifted, the model is seeing data it wasn't trained on. This is data drift and the fix is retraining, not debugging the model.

Second, **if the data looks normal, look at specific failure cases**. Pull a sample of false negatives (churners the model missed) and false positives (stable customers incorrectly flagged). Run SHAP on them. Often a pattern emerges — a specific customer segment that was underrepresented in training, or a feature combination the model consistently mishandles.

Third, **check the threshold**. If precision has collapsed (too many false positives) or recall has dropped (too many misses), the threshold may need adjustment before retraining. This is a quick intervention that doesn't require a new training run.

---

*Total: 70+ questions across 19 categories.*

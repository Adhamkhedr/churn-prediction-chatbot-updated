# Project Report -AI-Powered Churn Prediction Chatbot

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Business Context & Motivation](#2-business-context--motivation)
3. [Data Analysis (EDA)](#3-data-analysis-eda)
4. [Data Preprocessing & Feature Engineering](#4-data-preprocessing--feature-engineering)
5. [Model Selection & Training](#5-model-selection--training)
6. [Model Evaluation](#6-model-evaluation)
7. [Explainability (SHAP)](#7-explainability-shap)
8. [Solution Architecture](#8-solution-architecture)
9. [LLM Integration](#9-llm-integration)
10. [How This Meets Client Requirements](#10-how-this-meets-client-requirements)
11. [Limitations & Honest Assessment](#11-limitations--honest-assessment)
12. [Future Enhancements](#12-future-enhancements)

---

## 1. Executive Summary

This project delivers a Proof of Concept (PoC) for an AI-powered churn prediction system. It combines a tuned XGBoost classification model with a locally-hosted open-source LLM (Mistral 7B via Ollama) to create a chatbot that allows a marketing team to predict customer churn through natural language conversation.

The system achieves a **Recall of 81.3%** (correctly identifies 81 out of 100 customers who will churn) with a **ROC-AUC of 0.846**. It provides personalized, per-customer risk factor explanations using SHAP (SHapley Additive exPlanations), giving the marketing team not just a prediction but an understanding of *why* a customer is at risk.

The entire solution runs locally with no external API calls, meeting the client's requirement for open-source models and data privacy.

---

## 2. Business Context & Motivation

### The Cost of Churn

Customer acquisition costs 5 to 7 times more than customer retention. In the telecom industry, where competition is high and switching barriers are low, churn directly impacts revenue. Identifying at-risk customers before they leave allows the marketing and retention teams to intervene proactively -offering targeted discounts, contract upgrades, or service bundles.

### Why Predict Churn?

Traditional approaches rely on reactive analysis: a customer leaves, and the team investigates why after the fact. A predictive model flips this -it identifies customers who are *likely* to leave based on their current profile, enabling proactive retention outreach while there is still an opportunity to act.

### Why a Chatbot?

The marketing team is the end user. They are not data scientists. Requiring them to use dashboards, fill structured forms, or interact with a model through code would create friction and limit adoption. A chatbot interface allows them to describe a customer in their own words -the way they naturally think about their customers -and receive an understandable result.

### Why Open-Source and Local?

The client explicitly requires that the solution does not rely on closed-source models or third-party LLM APIs. This is a common requirement in enterprise environments for several reasons:
- **Data privacy**: Customer data never leaves the company's infrastructure.
- **No vendor lock-in**: The solution does not depend on OpenAI, Google, or any external provider.
- **Cost predictability**: No per-token API costs that scale with usage.
- **Compliance**: Meets data residency and regulatory requirements without additional configuration.

---

## 3. Data Analysis (EDA)

### Dataset Overview

The dataset contains **7,043 telecom customer records** with 21 variables covering demographics, account information, service subscriptions, and churn status. The target variable is `Churn` (Yes/No) -whether the customer left in the last period.

### Class Distribution

- **No Churn**: 5,174 customers (73.4%)
- **Churn**: 1,869 customers (26.5%)

This is a **moderately imbalanced** dataset. The imbalance is not severe enough to require oversampling techniques (like SMOTE), but it must be accounted for in the model -without handling it, the model would achieve 73% accuracy by simply predicting "No Churn" for everyone, which is useless.

### Key Findings

**1. Contract Type is a Major Separator**

*How we found it:* EDA Part 5 grouped all customers by Contract type and computed the churn rate for each group (churned / total in that group). A grouped bar chart was plotted showing churned vs non-churned counts per category. The churn rate spread (max rate − min rate) was computed and ranked across all categorical features.

*What we saw:*
- Month-to-month: 42.7% churn
- One year: 11.3% churn
- Two year: 2.8% churn
- Spread: **39.88 percentage points** — the strongest categorical predictor in the entire dataset

*Conclusion:* Customers with no long-term commitment can leave at any time with zero cost. Contract type became one of the 6 quick-mode features in the chatbot.

---

**2. Fiber Optic Internet Correlates with Higher Churn**

*How we found it:* Same churn rate spread analysis in EDA Part 5. Grouped customers by Internet_Service type and computed churn rate per group.

*What we saw:*
- Fiber optic: ~41.9% churn
- DSL: ~19.0% churn
- No internet: ~7.4% churn
- Spread: **34.49 percentage points** — second strongest categorical predictor

*Conclusion:* This is counterintuitive — fiber is a premium service. The explanation is that fiber optic customers pay more, and when premium-paying customers feel the service doesn't match the price, they are more likely to leave. Higher expectations lead to higher dissatisfaction when unmet.

---

**3. Tenure is the Strongest Numeric Predictor**

*How we found it:* EDA Part 4 computed the Pearson correlation between each numeric feature and Churn (after encoding Churn as 0/1). Boxplots were plotted showing the distribution of each numeric feature split by Churn=Yes vs Churn=No.

*What we saw:*
- tenure correlation with Churn: **−0.35** (strongest among numeric features)
- Median tenure for churners: ~10 months
- Median tenure for non-churners: ~38 months — a 28-month gap
- Monthly_Charges correlation: +0.19 (second strongest)

*Conclusion:* New customers churn far more than established ones. The longer someone stays, the more loyal they become. tenure became the #1 SHAP feature after training (importance 0.526).

---

**4. Short Tenure + High Charges = Danger Zone**

*How we found it:* EDA Part 6 performed a cross-feature quadrant analysis. Customers were split at the median tenure (29 months) and median Monthly_Charges ($64.76), creating four quadrants. The churn rate was computed for each quadrant.

*What we saw:*
- Low tenure + High charges: **58.03% churn** — highest risk segment
- Low tenure + Low charges: ~26% churn
- High tenure + High charges: ~29% churn
- High tenure + Low charges: ~5.5% churn — safest segment

*Conclusion:* New customers paying premium prices haven't yet developed loyalty. They represent the highest-risk segment and the primary target for proactive retention.

---

**5. Churn Drops Sharply with Tenure Over Time**

*How we found it:* EDA Part 9 split customers into 6-month tenure bands (0-12, 13-24, 25-36, 37-48, 49-60, 61-72 months) and computed the churn rate for each band.

*What we saw:*
- 0-12 months: **47.44% churn**
- 13-24 months: ~35% churn
- 25-36 months: ~25% churn
- 37-48 months: ~18% churn
- 49-60 months: ~11% churn
- 61-72 months: **6.61% churn**

*Conclusion:* The first year is the highest-risk window. Churn rate drops by more than 7x between new customers and long-term customers. This reinforced tenure as the top feature in the model.

---

**6. Support Services Reduce Churn**

*How we found it:* EDA Part 5 churn rate spread analysis for Online_Security and Tech_Support.

*What we saw:*
- No Online Security: ~42% churn | Has Online Security: ~15% churn → **34.36pp spread**
- No Tech Support: ~42% churn | Has Tech Support: ~15% churn → **34.23pp spread**

*Conclusion:* Support add-ons significantly reduce churn risk, likely because they increase perceived value and create switching costs. Both became part of the 6 quick-mode features in the chatbot.

---

**7. Total Charges is Redundant with Tenure**

*How we found it:* EDA Part 4 computed the Pearson correlation matrix between all numeric features. A heatmap was plotted.

*What we saw:*
- Correlation between Total_Charges and tenure: **0.83**
- Correlation between Total_Charges and Churn: −0.20 (weaker than tenure's −0.35)

*Conclusion:* A customer who has been with the company longer naturally accumulates higher total charges. They carry nearly the same signal. Including both gives the model the same information twice, adding noise rather than value. This was confirmed empirically: the Version A/B/C feature experiment showed Version C (dropping Total_Charges) performed best on Validation Recall.

---

**8. Three Features Have Near-Zero Predictive Power**

*How we found it:* EDA Part 5 churn rate spread analysis ranked all 16 categorical features. The bottom three were:

| Feature | Churn Rate Spread | Notes |
|---|---|---|
| gender | 0.76pp | Male 26.2% vs Female 26.9% — essentially identical |
| Phone_Service | 1.78pp | 90% of customers have it — near-constant feature |
| Dual (multiple lines) | 3.68pp | Minimal separation between groups |

*Conclusion:* These three features have no meaningful discriminative power. A 0.76pp difference between male and female churn rates is not a pattern — it is noise. All three were later confirmed by permutation importance on the validation set and sequential ablation testing: dropping all three together improved Recall from 0.8075 to 0.8209.

### How EDA Guided My Decisions

| EDA Finding | Decision Made |
|---|---|
| Total_Charges correlated with tenure (0.83) | Tested dropping it -Version C confirmed removal improves parsimony with no performance loss |
| Gender, Phone_Service, Dual are weak | Confirmed by permutation importance on validation set (near-zero Recall drop). Ablation test showed dropping all three together improved Recall from 0.8075 to 0.8209. Permanently dropped. |
| 11 rows have blank Total_Charges (all tenure=0) | Filled with 0 instead of dropping -these are new customers not yet billed, and 0 is the correct value |
| Class imbalance (73/27) | Used `scale_pos_weight=2.77` in XGBoost to upweight the minority class |
| Contract, tenure, internet type are strong separators | These became part of the "quick mode" features in the chatbot |
| Support services reduce churn | Online Security and Tech Support included in quick mode |

---

## 4. Data Preprocessing & Feature Engineering

### Data Cleaning

**Handling blank Total Charges**: 11 rows had blank (non-numeric) Total_Charges values. All 11 were customers with tenure = 0 -new customers who had not yet been billed. Rather than dropping these rows (losing valid data), I filled them with 0, which is the correct value for their situation.

**Column cleanup**: Stripped whitespace from column names. Dropped `customerID` as it is a unique identifier with no predictive value.

### Encoding Strategy

The encoding approach was chosen specifically for tree-based models (XGBoost):

| Column Type | Encoding | Reason |
|---|---|---|
| Binary (Yes/No) columns: Is_Married, Dependents, Paperless_Billing | OrdinalEncoder (No=0, Yes=1) | Two categories -ordinal encoding is sufficient and efficient |
| Multi-category columns: Internet_Service, Contract, Payment_Method | OneHotEncoder (drop first) | Multiple unordered categories -one-hot avoids imposing false ordinal relationships |
| Service columns: Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV, Streaming_Movies | OrdinalEncoder | Three categories (Yes/No/No internet service) -ordinal is sufficient for tree models which will learn the splits |
| Numeric: Senior_Citizen, tenure, Monthly_Charges | Passthrough (no transformation) | Tree-based models are scale-invariant -normalization and standardization are not needed and would not improve performance |

### Why No Scaling or Normalization

XGBoost is a tree-based model. Trees split on feature values using thresholds -the scale of a feature does not affect the split point selection. Whether tenure is measured in months (0-72) or standardized to mean 0 with standard deviation 1, the XGBoost tree finds the same optimal splits. Scaling is only necessary for distance-based models (KNN, SVM) or gradient-based models (Logistic Regression, Neural Networks) where feature magnitude affects optimization.

### Feature Engineering Experiments

**charges_per_tenure**: I engineered a feature dividing Total_Charges by tenure, representing the average monthly rate over a customer's lifetime. However, since the model already has Monthly_Charges directly, this was redundant.

**Three-version experiment**: To determine the optimal feature set, I trained and evaluated three versions:

| Version | Features Included | Result |
|---|---|---|
| A | All features including Total_Charges and charges_per_tenure | Baseline |
| B | Dropped Total_Charges (correlated with tenure at 0.83) | Similar performance, one less feature |
| C | Dropped Total_Charges and charges_per_tenure | Best -same or better performance with fewer features |

**Version C won.** Removing these redundant features did not hurt performance because the information they carry (how long a customer has been paying) is already captured by tenure. A simpler model with fewer features is preferable -it is less prone to overfitting, faster to train, and easier to explain.

### Weak Feature Removal

Weak feature removal followed a three-step methodology to ensure decisions were evidence-based and not influenced by the test set:

**Step 1 — EDA Pre-filter (churn rate spread):**
Before any model training, EDA Part 5 computed the churn rate spread for every categorical feature (max churn rate minus min churn rate across categories). Features with near-zero spread have no discriminative power. This flagged gender (0.76pp), Phone_Service (1.78pp), and Dual (3.68pp) as potentially weak — far below the next weakest feature (Is_Married at 13.29pp).

**Step 2 — Permutation Importance on Validation:**
After training a baseline model on the training set, permutation importance was computed on the validation set. For each feature, its values were randomly shuffled (destroying its signal) and the drop in Recall was measured. This was repeated 10 times per feature and averaged. Features where shuffling caused no meaningful Recall drop were flagged as weak. Importantly, this was done on the **validation set** (not the test set) to keep the test set untouched for final evaluation.

**Step 3 — Sequential Ablation Testing:**
To confirm the drop decisions, a sequential ablation test was run: starting from a baseline with all features, features were dropped one by one and performance was recorded at each step.

| Experiment | Recall | F1 | ROC-AUC |
|---|---|---|---|
| Baseline (all features) | 0.8075 | 0.6198 | 0.8441 |
| Drop gender | 0.8075 | 0.6198 | 0.8441 |
| Drop gender + Phone_Service | 0.8102 | 0.6209 | 0.8453 |
| Drop gender + Phone_Service + Dual | **0.8209** | **0.6253** | **0.8471** |

Dropping all three together improved Recall by 1.34 percentage points. Each feature was adding slight noise that collectively degraded performance. All three were permanently removed from the dataset in `preprocessing.py`.

**Total_Charges** was also dropped separately: it had a 0.83 correlation with tenure (nearly identical signal) and a permutation importance of 0.0000 (shuffling it caused zero Recall drop). The Version A/B/C numeric feature experiment confirmed Version C — dropping Total_Charges — as the best performing configuration.

The final model uses **15 input features** (down from the original 20 in the raw dataset), producing **19 encoded features** after one-hot encoding expands the multi-category columns.

---

## 5. Model Selection & Training

### Why XGBoost

I evaluated the task requirements against several model families:

| Model | Pros | Cons | Decision |
|---|---|---|---|
| **XGBoost** | Handles imbalanced data natively (scale_pos_weight), captures non-linear interactions, built-in feature importance, no scaling required, strong performance on tabular data | Requires hyperparameter tuning | **Selected** |
| Logistic Regression | Simple, interpretable, fast | Linear -cannot capture interactions like "short tenure AND high charges" | Used as baseline only |
| Random Forest | Similar strengths to XGBoost | Generally lower performance than boosted methods on structured data | Not selected |
| Neural Networks | Powerful for complex patterns | Overkill for 7,043 rows of tabular data, harder to interpret, requires more data | Not selected |
| SVM | Strong theoretical foundation | Requires scaling, slow on larger datasets, less interpretable | Not selected |

XGBoost was chosen because it is the standard choice for tabular classification tasks. It naturally handles the challenges in this data: class imbalance, mixed feature types, and non-linear relationships between features and the target.

### Training Configuration

**Three-Way Split**: The data was split into three sets to prevent data leakage — all intermediate decisions (feature selection, hyperparameter tuning) must be made without any exposure to the test set.

| Set | Size | Purpose |
|---|---|---|
| Train | 4,507 rows (64%) | Model training and hyperparameter tuning (via CV) |
| Validation | 1,127 rows (16%) | Permutation importance for feature drop decisions |
| Test | 1,409 rows (20%) | Final evaluation — touched exactly once |

All splits are stratified to maintain the 73/27 churn ratio. Fixed random_state=42 for reproducibility.

**Why three-way and not two-way?** With a two-way split, the same held-out set would be used for both intermediate decisions (which features to drop, which hyperparameters to use) and final reporting. Every time a decision is made using the test set, the model indirectly "sees" it, inflating reported performance. A three-way split ensures the test set is genuinely unseen.

**Class Imbalance Handling**: `scale_pos_weight = 2.77` (ratio of non-churners to churners in the training set). This tells XGBoost to weight misclassifications of the minority class (churners) 2.77 times more heavily, effectively upsampling without creating synthetic data. This was chosen over SMOTE or random oversampling because it is built into the algorithm, introduces no synthetic artifacts, and is computationally free.

**Cross-Validation**: 5-fold Stratified K-Fold. Each fold preserves the class distribution. This provides a reliable estimate of model performance by training and evaluating on 5 different train/validation splits, reducing the risk of an optimistic or pessimistic estimate from a single split.

### Hyperparameter Tuning

Default XGBoost hyperparameters (max_depth=6, learning_rate=0.3, n_estimators=100) caused overfitting on this relatively small dataset, resulting in the model performing **worse than a simple Logistic Regression** on Recall (0.679 vs 0.778).

I used **RandomizedSearchCV** with 50 iterations across 5 folds (250 total fits), optimizing for **Recall** -because in a churn prediction context, missing a customer who will churn (false negative) is more costly than incorrectly flagging a stable customer (false positive).

**Search Space:**

| Hyperparameter | Values Tested | Purpose |
|---|---|---|
| max_depth | 3, 4, 5, 6, 7 | Maximum tree depth -lower values reduce overfitting |
| n_estimators | 100, 200, 300, 500 | Number of trees -more trees with lower learning rate can improve generalization |
| learning_rate | 0.01, 0.05, 0.1, 0.2 | Step size per tree -lower values need more trees but generalize better |
| min_child_weight | 1, 3, 5, 7 | Minimum samples per leaf -higher values prevent the model from learning noise |
| subsample | 0.6, 0.7, 0.8, 0.9, 1.0 | Fraction of rows used per tree -adds randomness to reduce overfitting |
| colsample_bytree | 0.6, 0.7, 0.8, 0.9, 1.0 | Fraction of features used per tree -adds randomness and reduces feature co-dependence |
| gamma | 0, 0.1, 0.3, 0.5, 1.0 | Minimum loss reduction to make a split -higher values make the model more conservative |
| reg_alpha | 0, 0.01, 0.1, 0.5 | L1 regularization -promotes sparsity, can zero out irrelevant feature weights |
| reg_lambda | 0.5, 1.0, 1.5, 2.0 | L2 regularization -penalizes large weights, reduces overfitting |

**Best Parameters Found:**

| Parameter | Value | Interpretation |
|---|---|---|
| max_depth | 3 | Shallow trees — prevents overfitting on 7,043 rows |
| n_estimators | 500 | More trees to compensate for the very low learning rate |
| learning_rate | 0.01 | Very small steps — each tree makes minimal corrections, improving generalization |
| min_child_weight | 5 | Requires at least 5 samples per leaf — prevents learning from noise |
| subsample | 0.6 | Each tree sees only 60% of the data — strong regularization through randomness |
| colsample_bytree | 0.6 | Each tree uses 60% of features — reduces feature co-dependence |
| gamma | 1.0 | Minimum loss reduction required to make a split — makes the model conservative |
| reg_alpha | 0.5 | Moderate L1 regularization — promotes sparsity |
| reg_lambda | 2.0 | Strong L2 regularization — penalizes large weights, reduces overfitting |

The overall theme of the best parameters is **strong regularization**: shallow trees (max_depth=3), aggressive row subsampling (60%), slow learning rate (0.01), and minimum leaf size of 5. This combination prevents overfitting on a relatively small dataset while still capturing the important patterns.

---

## 6. Model Evaluation

### Final Metrics (Test Set)

| Metric | XGBoost (Tuned) | Logistic Regression (Baseline) | Winner |
|---|---|---|---|
| Recall | **0.8128** | 0.7834 | XGBoost |
| F1 Score | **0.6275** | 0.6162 | XGBoost |
| Precision | **0.5109** | 0.5078 | XGBoost |
| ROC-AUC | **0.8458** | 0.8377 | XGBoost |

After tuning, XGBoost outperforms the Logistic Regression baseline on all four metrics.

### Why Recall is the Primary Metric

In churn prediction, the cost of a **false negative** (missing a customer who will churn) far exceeds the cost of a **false positive** (flagging a stable customer):

- **False negative**: The customer leaves. Revenue is lost. Acquisition of a replacement costs 5-7x more.
- **False positive**: The marketing team reaches out to a customer who was not actually going to leave. The cost is minimal -a retention offer or a phone call.

Optimizing for Recall means the model catches as many actual churners as possible, even if some stable customers get flagged. This is the correct tradeoff for the business.

### Understanding the Metrics

- **Recall (0.813)**: Of all customers who actually churn, the model correctly identifies 81.3%. It misses 18.7% — these are churners who slip through.
- **Precision (0.511)**: Of all customers the model flags as churners, about 51% actually churn. The other 49% are false alarms. This sounds low, but in retention marketing, reaching out to a stable customer has minimal cost.
- **F1 Score (0.628)**: The harmonic mean of Precision and Recall. Provides a balanced view.
- **ROC-AUC (0.846)**: Measures the model's ability to distinguish between churners and non-churners across all thresholds. A score of 0.846 indicates good discriminative power.

### Threshold Tuning

The default decision threshold is 0.50 -if the predicted probability exceeds 50%, the model predicts "Churn." I tested thresholds from 0.20 to 0.60:

| Threshold | Recall | Precision | F1 |
|---|---|---|---|
| 0.20 | 0.9626 | 0.3818 | 0.5467 |
| 0.30 | 0.9198 | 0.4257 | 0.5821 |
| 0.40 | 0.8610 | 0.4667 | 0.6053 |
| 0.45 | 0.8449 | 0.4899 | 0.6202 |
| **0.50** | **0.8128** | **0.5109** | **0.6275** |
| 0.55 | 0.7594 | 0.5441 | 0.6339 |
| 0.60 | 0.7059 | 0.5665 | 0.6286 |

The threshold of 0.55 achieves marginally better F1 (0.6339 vs 0.6275) but at the cost of Recall dropping from 0.813 to 0.759. Since catching churners is the priority, I retained the 0.50 threshold to maintain higher Recall.

### Overfitting Check

I compared model performance on the training set versus the test set:

| Metric | Train | Test | Gap |
|---|---|---|---|
| Recall | ~0.827 | 0.813 | ~1.4% |
| ROC-AUC | ~0.862 | 0.846 | ~1.6% |

A gap of approximately 1.5% between train and test performance indicates the model generalizes well. Gaps above 5-10% would suggest overfitting. The strong regularization from hyperparameter tuning (shallow trees, subsampling, low learning rate) is doing its job.

### Honest Assessment

- **Recall of 0.813 is good** for this dataset and task. It means the system catches 4 out of 5 at-risk customers.
- **Precision of 0.511 is the main weakness.** About half of the customers flagged as churners will not actually churn. For a PoC, this is acceptable — the marketing team treats these as leads for outreach, not automatic actions.
- **These scores are typical** for the Telco Customer Churn dataset with similar approaches in published benchmarks.
- Further improvement would require either additional data, external features (e.g., call center interactions, usage patterns), or more sophisticated techniques (deep learning, time-series modeling), which are outside the scope of this PoC.

---

## 7. Explainability (SHAP)

### Why Explainability Matters

A prediction without an explanation has limited business value. Telling the marketing team "this customer has an 85% churn risk" is useful, but telling them "this customer has an 85% churn risk **because they have short tenure, no online security, and fiber optic internet**" is actionable. The team can then target their retention effort: offer a security bundle, suggest a contract upgrade, or assign a dedicated support agent.

### What is SHAP?

SHAP (SHapley Additive exPlanations) is a method grounded in game theory that explains individual predictions. For each prediction, SHAP assigns every feature a value representing how much it pushed the prediction up or down relative to the average. A positive SHAP value pushes toward churn; a negative SHAP value pushes toward staying.

### Global vs Per-Customer SHAP

**Global SHAP** (used during model analysis) shows which features are most important across the entire dataset. The global ranking for the model:

1. tenure — 0.5258
2. Contract (Two year) — 0.4265
3. Fiber Optic Internet — 0.2954
4. Online Security — 0.2577
5. Tech Support — 0.2066
6. Contract (One year) — 0.1924
7. Monthly Charges — 0.1681
8. Electronic Check Payment — 0.1662
9. Paperless Billing — 0.1074
10. Online Backup — 0.0687

**Per-customer SHAP** (used in the chatbot) shows which features matter for **one specific customer**. This is critical because the same feature can push in opposite directions:

- For a customer with **2 months tenure**: tenure has a positive SHAP value (increasing churn risk)
- For a customer with **60 months tenure**: tenure has a negative SHAP value (reducing churn risk)

### How SHAP is Used in the Chatbot

When the model makes a prediction, it computes SHAP values for that specific customer using `shap.TreeExplainer`. It then selects the top 3 features by absolute SHAP value and display them with their direction:

- "**Tenure (Months)** -increasing churn risk" (short tenure is hurting this customer)
- "**Two Year Contract** -reducing churn risk" (the contract is keeping this customer loyal)

This provides the marketing team with a specific, actionable explanation for every prediction -not a generic list.

### Why SHAP Over Basic Feature Importance

XGBoost provides built-in feature importance based on how often a feature is used in tree splits (gain). However, this gives the same ranking for every prediction -it is a property of the model, not of the individual customer. SHAP is superior because:

1. It is **personalized** -different customers get different explanations
2. It provides **direction** -not just "this feature matters" but "this feature is pushing toward/away from churn"
3. It accounts for **feature interactions** -gain-based importance does not
4. It revealed that **tenure is the #1 driver** (SHAP rank 1), whereas basic feature importance ranked it 6th -showing that SHAP provides a more accurate picture

---

## 8. Solution Architecture

The system consists of four main components connected in a pipeline:

```
┌──────────────────────────────────────────────────────────────┐
│                        USER (Marketing Team)                  │
│                     Streamlit Chat Interface                  │
└──────────────┬───────────────────────────────┬───────────────┘
               │ HTTP Request                  │ HTTP Response
               ▼                               │
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                          │
│              /chat    /reset    /health                       │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│                    Chatbot Pipeline                           │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │  Ollama /   │    │   Slot      │    │   XGBoost       │  │
│  │  Mistral    │───▶│   Filling   │───▶│   Model +       │  │
│  │  (Extract)  │    │   & Valid.  │    │   SHAP          │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                                                              │
│  User message ──▶ Structured data ──▶ Prediction + Factors   │
└──────────────────────────────────────────────────────────────┘
```

### Component Details

**Streamlit UI** (`ui/app.py`): The frontend chat interface. Sends user messages to the API and renders responses. Includes session management (New Session button), a typing indicator, and color-coded prediction result boxes (green for low risk, red for high risk). Chosen for its simplicity -a Python-native framework that requires no frontend development expertise.

**FastAPI Backend** (`api/main.py`): A lightweight REST API with three endpoints. Manages sessions in-memory (each chat session maintains its own state). Chosen for its performance, automatic OpenAPI documentation, and minimal boilerplate. The API can be consumed by any HTTP client, making the solution extensible beyond the Streamlit UI.

**Chatbot Pipeline** (`chatbot/pipeline.py`): The core logic. Handles the conversation flow:
1. Sends user messages to Mistral for structured data extraction
2. Validates extracted values against known categories
3. Tracks which fields have been collected (slot filling)
4. Triggers prediction when enough data is available
5. Computes per-customer SHAP explanations
6. Manages quick mode / full mode transitions

**Ollama + Mistral** (external service): Runs the Mistral 7B language model locally. Receives prompts from the pipeline and returns structured JSON extractions or conversational follow-up questions. Communicates via HTTP on localhost:11434.

**XGBoost Model** (`model/churn_pipeline.pkl`): The trained classification pipeline including preprocessing (encoding) and the XGBoost classifier. Loaded once at startup for fast inference.

### Technology Choices

| Component | Choice | Reasoning |
|---|---|---|
| ML Framework | XGBoost + scikit-learn | Industry standard for tabular classification; scikit-learn Pipeline ensures preprocessing and model are bundled together |
| LLM | Mistral 7B via Ollama | Open-source, runs locally, no API costs, meets client requirements |
| API Framework | FastAPI | Lightweight, high-performance, auto-generates API documentation |
| UI Framework | Streamlit | Python-native, rapid development, no frontend expertise needed |
| Explainability | SHAP TreeExplainer | Fast for tree models, theoretically grounded, per-prediction explanations |
| Model Persistence | joblib (pkl) | Standard for scikit-learn pipelines, includes all preprocessing metadata |

---

## 9. LLM Integration

### Role of the LLM

The LLM (Mistral 7B) serves as a **natural language interface** between the marketing team and the ML model. It performs two specific tasks:

1. **Extraction**: Parse unstructured user messages into structured field values (e.g., "been with us for 3 months" → `tenure: 3`)
2. **Conversation**: Generate natural follow-up questions to collect missing fields

The LLM does **not** make the churn prediction itself -that is handled entirely by the XGBoost model. The LLM is purely an interface layer.

### Extraction Process

When the user sends a message, the pipeline builds a structured prompt instructing Mistral to extract specific fields. The prompt includes:
- The list of fields still needed
- Valid values for categorical fields
- Explicit instructions to return `null` for anything not mentioned
- A JSON format specification

Example: If the user says *"Customer on a month-to-month contract with fiber optic internet, paying $85"*, Mistral returns:
```json
{"Contract": "Month-to-month", "Internet_Service": "Fiber optic", "Monthly_Charges": 85, "tenure": null, "Tech_Support": null, "Online_Security": null}
```

### Anti-Hallucination Safeguards

Small language models like Mistral 7B are prone to hallucination -generating plausible but fabricated values. In a prediction system, hallucinated inputs lead to incorrect predictions. I implemented two layers of protection:

**1. Prompt Engineering**: The extraction prompt repeatedly emphasizes using `null` for unmentioned fields. The wording was iteratively refined to minimize fabrication.

**2. Keyword Validation**: After Mistral returns extracted values, a post-processing layer checks whether the user's message actually contains words related to each extracted field. For example, if Mistral extracts `"Tech_Support": "No"` but the user never mentioned "tech", "support", or related keywords, the extraction is rejected. This catches the most common hallucination pattern where Mistral fills in default values for fields the user never discussed.

### Quick Mode vs Full Mode
The chatbot implements a two-stage collection process based on SHAP feature importance:

**Quick Mode** collects the 6 most influential features first:
- tenure, Contract, Monthly_Charges, Internet_Service, Tech_Support, Online_Security

These 6 features were selected based on three converging sources of evidence: EDA Part 5 churn rate spread (Contract 39.88pp, Internet_Service 34.49pp, Online_Security 34.36pp, Tech_Support 34.23pp — the top 4 categorical predictors), EDA Part 4 Pearson correlation (tenure −0.35 and Monthly_Charges +0.19 — the top 2 numeric predictors), and global SHAP importance which confirmed all 6 appear in the top 7 features by mean absolute SHAP value across the test set. If the resulting prediction is confident — below 20% or above 80% probability — the result is returned immediately.

**Full Mode** is triggered only when the quick prediction falls in the uncertain range (20-80%). The chatbot then asks for the remaining 9 features to refine the prediction. This design choice reduces friction for clear-cut cases while maintaining accuracy for borderline ones.

### End-to-End Message Flow

The following describes exactly what happens from the moment a marketing team member sends a message to when a prediction is returned.

**Session creation**: When the first message arrives at `main.py`, it checks whether the `session_id` already exists in the sessions dictionary. If not, a new `Session` object is created with all 15 slots set to `None`, mode set to `quick`, and `prediction_made` set to `False`. For every message after that in the same conversation, the existing session is reused — the team member won't provide all customer data in one message, so the session persists state across multiple turns.

**Prediction lock check**: The first thing `handle_message()` does is check whether `prediction_made` is `True`. If so, the session is locked and a message is immediately returned asking the team member to start a new session. This ensures each session produces exactly one prediction per customer.

**Extraction**: `build_extraction_prompt()` looks at which slots are filled and which are missing, converts the missing ones to human-readable descriptions using `FIELD_DESCRIPTIONS`, and assembles a prompt instructing Mistral to extract field values from the message as JSON — with null for anything not explicitly mentioned. This prompt is sent to Mistral via `call_ollama()`, which posts it to Ollama's local HTTP endpoint (`http://localhost:11434/api/generate`) and waits for the response.

**Validation**: The raw text response goes through `extract_slots_from_response()` which does three things in order: converts the JSON text string to a Python dictionary using `json.loads()`, runs a hallucination check that rejects any extracted field whose keywords don't appear in the original message, and validates types and values (numbers coerced to float, categoricals matched case-insensitively against the valid values whitelist). Only values passing all three checks are saved to the session slots.

**Check and respond**: `all_required_filled()` checks whether all required slots for the current mode are filled. If not, `build_conversation_prompt()` tells Mistral which fields are already collected and which are still missing, Mistral generates a natural friendly reply, and that reply is sent back along with an acknowledgment of what was just extracted. The conversation loops until all required slots are filled.

**Prediction**: When all required slots are filled, `make_prediction()` is called. It builds a one-row DataFrame from session slots, fills any uncollected columns with defaults (median or mode from training data saved in the pkl), runs `pipeline.predict_proba()` to get the churn probability, and runs SHAP to find the top 3 features driving that specific prediction. Each SHAP value is paired with a direction (positive = increasing churn risk, negative = reducing churn risk) and mapped to a human-readable name.

**Confidence check**: If mode is quick and the probability falls between 0.20 and 0.80, the session switches to full mode and the conversation continues asking for the remaining 9 slots. If the probability is outside that range or full mode is already complete, `build_explanation()` converts the numbers into a plain English message with risk level, probability percentage, top 3 SHAP factors, and a recommendation. `prediction_made` is set to `True` and the result is returned.

```
Message arrives → session created/retrieved
        │
        ▼
prediction_made? → Yes → block, return "start new session"
        │ No
        ▼
build_extraction_prompt() → call_ollama() → Mistral returns JSON text
        │
        ▼
extract_slots_from_response()
  → json.loads(): text → dictionary
  → hallucination check: keywords present in message?
  → type & value validation
  → save clean values to session slots
        │
        ▼
all_required_filled()?
        │ No                              │ Yes
        ▼                                 ▼
build_conversation_prompt()         make_prediction()
→ call_ollama()                       → build row (slots + defaults)
→ friendly reply to team member       → pipeline.predict_proba()
→ loop                                → SHAP top 3 factors
                                           │
                                     quick mode + 0.20–0.80?
                                       │ Yes          │ No
                                       ▼              ▼
                                  switch to     build_explanation()
                                  full mode     prediction_made = True
                                  loop again    return result
```

### Performance Considerations

Mistral 7B running on CPU produces responses in 30-60 seconds. Each user message requires 1-2 LLM calls (extraction + optional follow-up generation). This latency is inherent to running a 7-billion parameter model on CPU hardware and is not a limitation of the pipeline implementation. With GPU acceleration, response times drop to 2-5 seconds.

---

## 10. How This Meets Client Requirements

| Client Requirement | How It Is Addressed |
|---|---|
| **Develop a churn classification model** | XGBoost model with leakage-safe three-way split, hyperparameter tuning, permutation importance for feature selection, and SHAP explainability. Recall of 0.813, ROC-AUC of 0.846. |
| **Marketing team interacts through an LLM-powered chatbot** | Streamlit chat interface with natural language input. No forms, no technical knowledge required. |
| **Open-source models, no closed-source or third-party LLMs** | Mistral 7B via Ollama -fully open-source, runs locally. No data leaves the infrastructure. |
| **Translates marketing questions into structured data** | Mistral extracts structured fields from natural language with anti-hallucination safeguards. |
| **Presents results in a clear and understandable format** | Probability percentage, risk level (LOW/MODERATE/HIGH), personalized top 3 risk factors with direction, and actionable recommendations. |
| **Deploy as an API** | FastAPI with /chat, /reset, and /health endpoints. Can be consumed by any HTTP client. |
| **Diagram showing architecture** | Architecture diagram provided showing component interactions and data flow. |
| **Documentation of approach** | This report covers all technical and business decisions with justification. |
| **Simple solution** | Four components (UI, API, Pipeline, LLM), single `python run.py` command to start, standard Python libraries. |

---

## 11. Limitations & Honest Assessment

### Model Limitations

- **Precision is 50%**: Half of flagged customers will not actually churn. This means the retention team will spend effort on some customers who were going to stay anyway. For a PoC, this is acceptable -the cost of outreach is low compared to the cost of losing a customer.
- **Static dataset**: The model is trained on a fixed snapshot of customer data. It does not learn from new churn events over time without retraining.
- **Single domain**: Trained specifically on telecom customer data. The model and feature engineering would need to be adapted for other industries.
- **No temporal features**: The dataset does not include time-series information (e.g., usage trends, recent support tickets). Adding these could improve predictions.

### LLM Limitations

- **CPU latency**: 30-60 seconds per response on CPU. This is noticeable in a conversational interface. GPU deployment would resolve this.
- **Occasional hallucination**: Despite safeguards, Mistral may occasionally extract values the user did not explicitly state. The keyword validation layer catches most cases but not all.
- **Model size**: Mistral 7B requires approximately 4.4GB of disk space and significant RAM. This is manageable on most modern machines but may be a consideration for constrained environments.

### Scope Limitations

- **No batch processing**: The current interface handles one customer at a time. A marketing team analyzing hundreds of customers would need a batch endpoint.
- **No integration with existing systems**: The chatbot requires manual input of customer data. Integration with CRM or billing systems would eliminate this friction.
- **Single language**: The chatbot operates in English only.

---

## 12. Future Enhancements

### Recommendation Engine

Currently, the chatbot provides a generic recommendation ("Proactive retention measures are advised"). A natural extension would be an intelligent recommendation engine that maps specific risk factors to specific retention offers:

- If "No Online Security" is a top risk factor → Recommend a security bundle at a discounted rate
- If "Month-to-month Contract" is a risk factor → Offer a one-year contract with a loyalty discount
- If "Short Tenure" is a risk factor → Trigger a new customer onboarding program

This would make the chatbot not just diagnostic ("this customer is at risk") but prescriptive ("here is what to do about it"), significantly increasing its business value.

### Customer Self-Service

A version of the chatbot could be deployed for customers themselves. Instead of the marketing team checking risk, the customer interacts with the chatbot and receives proactive offers. For example: "Based on your account, you might benefit from a security bundle -here is a 30% discount offer." This turns a reactive retention process into a proactive customer engagement tool.

### Batch Prediction Dashboard

Add an endpoint that accepts a CSV of customer records and returns churn predictions for all of them in a single operation. This would allow the marketing team to score their entire customer base periodically, segment by risk level, and plan targeted campaigns.

### Real-Time Data Integration

Connect the system to the company's CRM, billing, and support platforms. Instead of the marketing team manually describing a customer, the chatbot could pull live data: "Check customer #12345" → automatically retrieves their current account details and makes a prediction. This eliminates manual input errors and dramatically speeds up the workflow.

### Model Retraining Pipeline

Implement automated model retraining on a scheduled basis (e.g., monthly or quarterly) as new churn data becomes available. This ensures the model stays current with evolving customer behavior patterns and does not degrade over time.

### GPU Deployment

Deploying Ollama with GPU acceleration would reduce LLM response times from 30-60 seconds to 2-5 seconds, making the chatbot experience feel near-instant. This is a simple infrastructure change with a significant user experience impact.

### A/B Testing Framework

Build a framework to measure the effectiveness of retention strategies. When the system recommends an action for a high-risk customer, track whether the action was taken and whether the customer ultimately churned. This closes the feedback loop and allows continuous improvement of both the model and the retention strategies.

### Multi-Channel Deployment

Integrate the chatbot into tools the marketing team already uses: Slack, Microsoft Teams, or internal portals. This reduces adoption friction -the team does not need to open a separate application to check churn risk.

### Customer Segmentation

Use the model's predictions and SHAP values to automatically cluster customers into risk segments (e.g., "High Risk - Contract Issue", "High Risk - Service Dissatisfaction", "Moderate Risk - Price Sensitive"). This enables more targeted marketing campaigns than individual customer lookups.


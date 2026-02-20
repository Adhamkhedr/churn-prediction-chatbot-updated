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

The system achieves a **Recall of 81.0%** (correctly identifies 81 out of 100 customers who will churn) with a **ROC-AUC of 0.845**. It provides personalized, per-customer risk factor explanations using SHAP (SHapley Additive exPlanations), giving the marketing team not just a prediction but an understanding of *why* a customer is at risk.

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

Month-to-month contracts have a churn rate of approximately 42%, compared to just 11% for one-year contracts and 3% for two-year contracts. This makes intuitive sense -customers with no commitment can leave at any time with zero cost.

**2. Fiber Optic Internet Correlates with Higher Churn**

Customers with fiber optic internet churn at roughly 42%, compared to approximately 19% for DSL. This is counterintuitive at first -fiber is a premium service. However, fiber optic customers tend to pay more, and when premium-paying customers feel the service does not match the price, they are more likely to leave. Higher expectations lead to higher dissatisfaction when unmet.

**3. Short Tenure + High Charges = Danger Zone**

Customers who have been with the company for less than 12 months and pay more than $70/month have a churn rate of approximately 58%. These are new customers paying premium prices who have not yet developed loyalty. They represent the highest-risk segment.

**4. Long Tenure + Low Charges = Safe Zone**

Customers with more than 36 months of tenure paying under $50/month churn at only 5.5%. These are established, cost-effective customers -the most stable segment.

**5. Support Services Reduce Churn**

Customers without tech support churn at approximately 42% versus 15% with it. Similarly, customers without online security churn at approximately 42% versus 15% with it. The presence of support add-ons significantly reduces churn risk, likely because they increase the perceived value and create switching costs.

**6. Total Charges is Redundant with Tenure**

Total Charges and tenure have a correlation of 0.83. This makes sense -a customer who has been with the company longer will naturally have higher total charges. Including both in the model introduces multicollinearity without adding new information.

**7. Three Features Have Near-Zero Predictive Power**

- **Gender**: Less than 1 percentage point difference in churn rate between male and female customers.
- **Phone Service**: Less than 4 percentage points difference. Nearly all customers (90%) have phone service, making this feature almost constant.
- **Dual (multiple lines)**: Less than 4 percentage points difference. Offers minimal discriminative power.

These findings were later confirmed by both XGBoost feature importance and SHAP analysis, where all three ranked at the bottom.

### How EDA Guided My Decisions

| EDA Finding | Decision Made |
|---|---|
| Total_Charges correlated with tenure (0.83) | Tested dropping it -Version C confirmed removal improves parsimony with no performance loss |
| Gender, Phone_Service, Dual are weak | Tested dropping them -SHAP confirmed near-zero impact, dropped from final model |
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

After training the model, I conducted a controlled experiment: train with and without the three weak features identified in EDA (gender, Phone_Service, Dual).

| Metric | With Weak Features | Without | Difference |
|---|---|---|---|
| Recall | 0.8102 | 0.8102 | 0.0000 |
| F1 | 0.6209 | 0.6232 | +0.0023 |
| Precision | 0.5033 | 0.5059 | +0.0026 |
| ROC-AUC | 0.8453 | 0.8451 | -0.0002 |

The results confirmed what EDA and SHAP both predicted: these features carry virtually no signal. Dropping them slightly improved F1 and Precision while maintaining Recall. The final model uses **15 input features** (down from the original 20), producing **19 encoded features** after preprocessing.

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

**Train/Test Split**: 80% train (5,634 rows), 20% test (1,409 rows). Stratified split to maintain the 73/27 class ratio in both sets. Fixed random_state=42 for reproducibility.

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
| max_depth | 3 | Shallow trees -prevents overfitting on 7,043 rows |
| n_estimators | 300 | More trees to compensate for the low learning rate |
| learning_rate | 0.01 | Very small steps -each tree makes minimal corrections, improving generalization |
| min_child_weight | 5 | Requires at least 5 samples per leaf -prevents learning from noise |
| subsample | 0.6 | Each tree sees only 60% of the data -strong regularization through randomness |
| colsample_bytree | 0.8 | Each tree uses 80% of features -reduces feature co-dependence |
| gamma | 0 | No minimum loss requirement -the other regularization parameters are sufficient |
| reg_alpha | 0.1 | Light L1 regularization |
| reg_lambda | 0.5 | Moderate L2 regularization |

The overall theme of the best parameters is **strong regularization**: shallow trees (max_depth=3), aggressive row subsampling (60%), slow learning rate (0.01), and minimum leaf size of 5. This combination prevents overfitting on a relatively small dataset while still capturing the important patterns.

---

## 6. Model Evaluation

### Final Metrics (Test Set)

| Metric | XGBoost (Tuned) | Logistic Regression (Baseline) | Winner |
|---|---|---|---|
| Recall | **0.8102** | 0.7781 | XGBoost |
| F1 Score | **0.6209** | 0.6148 | XGBoost |
| Precision | 0.5033 | **0.5059** | LR (marginal) |
| ROC-AUC | **0.8453** | 0.8389 | XGBoost |

After tuning, XGBoost outperforms the Logistic Regression baseline on all metrics except Precision (where the difference is negligible at 0.26 percentage points).

### Why Recall is the Primary Metric

In churn prediction, the cost of a **false negative** (missing a customer who will churn) far exceeds the cost of a **false positive** (flagging a stable customer):

- **False negative**: The customer leaves. Revenue is lost. Acquisition of a replacement costs 5-7x more.
- **False positive**: The marketing team reaches out to a customer who was not actually going to leave. The cost is minimal -a retention offer or a phone call.

Optimizing for Recall means the model catches as many actual churners as possible, even if some stable customers get flagged. This is the correct tradeoff for the business.

### Understanding the Metrics

- **Recall (0.810)**: Of all customers who actually churn, the model correctly identifies 81%. It misses 19% -these are churners who slip through.
- **Precision (0.503)**: Of all customers the model flags as churners, about 50% actually churn. The other 50% are false alarms. This sounds low, but in retention marketing, reaching out to a stable customer has minimal cost.
- **F1 Score (0.621)**: The harmonic mean of Precision and Recall. Provides a balanced view.
- **ROC-AUC (0.845)**: Measures the model's ability to distinguish between churners and non-churners across all thresholds. A score of 0.845 indicates good discriminative power.

### Threshold Tuning

The default decision threshold is 0.50 -if the predicted probability exceeds 50%, the model predicts "Churn." I tested thresholds from 0.20 to 0.60:

| Threshold | Recall | Precision | F1 |
|---|---|---|---|
| 0.20 | 0.9519 | 0.3579 | 0.5202 |
| 0.30 | 0.9037 | 0.4126 | 0.5665 |
| 0.40 | 0.8610 | 0.4573 | 0.5974 |
| 0.45 | 0.8396 | 0.4797 | 0.6104 |
| **0.50** | **0.8102** | **0.5033** | **0.6209** |
| 0.55 | 0.7594 | 0.5345 | 0.6275 |
| 0.60 | 0.7005 | 0.5686 | 0.6278 |

The threshold of 0.55 achieves marginally better F1 (0.6275 vs 0.6209) but at the cost of Recall dropping from 0.810 to 0.759. Since catching churners is the priority, I retained the 0.50 threshold to maintain higher Recall.

### Overfitting Check

I compared model performance on the training set versus the test set:

| Metric | Train | Test | Gap |
|---|---|---|---|
| Recall | ~0.825 | 0.810 | ~1.5% |
| ROC-AUC | ~0.860 | 0.845 | ~1.5% |

A gap of approximately 1.5% between train and test performance indicates the model generalizes well. Gaps above 5-10% would suggest overfitting. The strong regularization from hyperparameter tuning (shallow trees, subsampling, low learning rate) is doing its job.

### Honest Assessment

- **Recall of 0.810 is good** for this dataset and task. It means the system catches 4 out of 5 at-risk customers.
- **Precision of 0.503 is the main weakness.** Half of the customers flagged as churners will not actually churn. For a PoC, this is acceptable -the marketing team treats these as leads for outreach, not automatic actions.
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

1. tenure -0.4817
2. Contract (Two year) -0.4437
3. Online Security -0.2528
4. Fiber Optic Internet -0.2313
5. Tech Support -0.2200
6. Contract (One year) -0.1797
7. Monthly Charges -0.1598
8. Electronic Check Payment -0.1116
9. Paperless Billing -0.0816
10. Streaming Movies -0.0658

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

These 6 features account for the majority of the model's predictive power (as confirmed by SHAP analysis). If the resulting prediction is confident -below 20% or above 80% probability -the result is returned immediately.

**Full Mode** is triggered only when the quick prediction falls in the uncertain range (20-80%). The chatbot then asks for the remaining 9 features to refine the prediction. This design choice reduces friction for clear-cut cases while maintaining accuracy for borderline ones.

### Performance Considerations

Mistral 7B running on CPU produces responses in 30-60 seconds. Each user message requires 1-2 LLM calls (extraction + optional follow-up generation). This latency is inherent to running a 7-billion parameter model on CPU hardware and is not a limitation of the pipeline implementation. With GPU acceleration, response times drop to 2-5 seconds.

---

## 10. How This Meets Client Requirements

| Client Requirement | How It Is Addressed |
|---|---|
| **Develop a churn classification model** | XGBoost model with hyperparameter tuning, cross-validation, and SHAP explainability. Recall of 0.810, ROC-AUC of 0.845. |
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

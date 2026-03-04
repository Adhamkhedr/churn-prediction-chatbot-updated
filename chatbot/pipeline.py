import json
import os
import sys
import requests
import joblib
import pandas as pd
import numpy as np
import shap

# Add model directory to path so pickle can find FeatureEngineer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from preprocessing import FeatureEngineer  # noqa: F401 — required for pickle

# ──────────────────────────────────────────────
# 1. LOAD MODEL
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'churn_pipeline.pkl')
model_data = joblib.load(MODEL_PATH)

pipeline = model_data['pipeline']
feature_names = model_data['feature_names']
defaults = model_data['defaults']
input_columns = model_data['input_columns']

# Initialize SHAP explainer for per-customer explanations
xgb_model = pipeline.named_steps['model']
explainer = shap.TreeExplainer(xgb_model)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# ──────────────────────────────────────────────
# 2. SLOT DEFINITIONS
# ──────────────────────────────────────────────
QUICK_SLOTS = ['tenure', 'Contract', 'Monthly_Charges', 'Internet_Service',
               'Tech_Support', 'Online_Security']

ALL_SLOTS = list(input_columns)

# Valid values for categorical fields (for validation)
# Note: gender, Phone_Service, Dual were dropped — EDA showed <4pp churn
# separation and both feature importance + SHAP confirmed near-zero impact.
VALID_VALUES = {
    'Senior_Citizen': [0, 1],
    'Is_Married': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'Internet_Service': ['DSL', 'Fiber optic', 'No'],
    'Online_Security': ['Yes', 'No', 'No internet service'],
    'Online_Backup': ['Yes', 'No', 'No internet service'],
    'Device_Protection': ['Yes', 'No', 'No internet service'],
    'Tech_Support': ['Yes', 'No', 'No internet service'],
    'Streaming_TV': ['Yes', 'No', 'No internet service'],
    'Streaming_Movies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'Paperless_Billing': ['Yes', 'No'],
    'Payment_Method': ['Electronic check', 'Mailed check',
                       'Bank transfer (automatic)', 'Credit card (automatic)'],
}

# Human-readable field descriptions for the LLM
FIELD_DESCRIPTIONS = {
    'Senior_Citizen': 'senior citizen status (0 for No, 1 for Yes)',
    'Is_Married': 'marital status (Yes/No)',
    'Dependents': 'has dependents/children (Yes/No)',
    'tenure': 'number of months with the company (a number)',
    'Internet_Service': 'internet service type (DSL/Fiber optic/No)',
    'Online_Security': 'has online security (Yes/No/No internet service)',
    'Online_Backup': 'has online backup (Yes/No/No internet service)',
    'Device_Protection': 'has device protection (Yes/No/No internet service)',
    'Tech_Support': 'has tech support (Yes/No/No internet service)',
    'Streaming_TV': 'has streaming TV (Yes/No/No internet service)',
    'Streaming_Movies': 'has streaming movies (Yes/No/No internet service)',
    'Contract': 'contract type (Month-to-month/One year/Two year)',
    'Paperless_Billing': 'has paperless billing (Yes/No)',
    'Payment_Method': 'payment method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))',
    'Monthly_Charges': 'monthly charges amount (a number)',
}

# Human-readable feature name mapping for risk factor explanations
FEATURE_DISPLAY_NAMES = {
    'Is_Married': 'Marital Status',
    'Dependents': 'Has Dependents',
    'Paperless_Billing': 'Paperless Billing',
    'Internet_Service_Fiber optic': 'Fiber Optic Internet',
    'Internet_Service_No': 'No Internet Service',
    'Contract_One year': 'One Year Contract',
    'Contract_Two year': 'Two Year Contract',
    'Payment_Method_Credit card (automatic)': 'Credit Card Payment',
    'Payment_Method_Electronic check': 'Electronic Check Payment',
    'Payment_Method_Mailed check': 'Mailed Check Payment',
    'Online_Security': 'Online Security',
    'Online_Backup': 'Online Backup',
    'Device_Protection': 'Device Protection',
    'Tech_Support': 'Tech Support',
    'Streaming_TV': 'Streaming TV',
    'Streaming_Movies': 'Streaming Movies',
    'Senior_Citizen': 'Senior Citizen',
    'tenure': 'Tenure (Months)',
    'Monthly_Charges': 'Monthly Charges',
}


# ──────────────────────────────────────────────
# 3. SESSION STATE
# ──────────────────────────────────────────────
class Session:
    def __init__(self):
        self.slots = {col: None for col in ALL_SLOTS}
        self.mode = 'quick'  # 'quick' or 'full'
        self.conversation_history = []
        self.prediction_made = False

    def filled_slots(self):
        return {k: v for k, v in self.slots.items() if v is not None}

    def missing_slots(self, mode=None):
        mode = mode or self.mode
        target_slots = QUICK_SLOTS if mode == 'quick' else ALL_SLOTS
        return [s for s in target_slots if self.slots[s] is None]

    def all_required_filled(self):
        return len(self.missing_slots()) == 0


# ──────────────────────────────────────────────
# 4. LLM INTERACTION
# ──────────────────────────────────────────────
def call_ollama(prompt):
    """Call Ollama API with Mistral model."""
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }, timeout=300)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Make sure Ollama is running (ollama serve)."
    except Exception as e:
        return f"ERROR: {str(e)}"


def build_extraction_prompt(user_message, session):
    """Build a prompt that asks the LLM to extract field values from user message."""
    missing = session.missing_slots()
    filled = session.filled_slots()

    missing_descriptions = "\n".join(
        f"  - {field}: {FIELD_DESCRIPTIONS[field]}"
        for field in missing
    )

    filled_info = ""
    if filled:
        filled_lines = "\n".join(f"  - {field}: {val}" for field, val in filled.items())
        filled_info = f"\nAlready collected:\n{filled_lines}\n"

    prompt = f"""Extract customer data from the user message below.

RULES:
1. ONLY extract values the user EXPLICITLY said. If a field is NOT mentioned, you MUST set it to null.
2. Do NOT guess, infer, or assume ANY values. When in doubt, use null.
3. For numeric fields (tenure, Monthly_Charges), extract the exact number stated.
4. For categorical fields, match to the closest valid option.
5. Return ONLY null for fields not clearly stated in the user message.
{filled_info}
Fields to extract (set to null if not mentioned):
{missing_descriptions}

User message: "{user_message}"

Return ONLY a JSON object. Use null for any field not explicitly mentioned.
Example: {{"tenure": 12, "Contract": "One year", "Monthly_Charges": null, "Internet_Service": null, "Tech_Support": null, "Online_Security": null}}

JSON:"""
    return prompt


def build_conversation_prompt(session):
    """Build a prompt that asks the LLM to conversationally request missing fields."""
    missing = session.missing_slots()
    filled = session.filled_slots()

    if not missing:
        return None

    missing_descriptions = "\n".join(
        f"  - {FIELD_DESCRIPTIONS[field]}"
        for field in missing
    )

    filled_info = ""
    if filled:
        filled_lines = "\n".join(f"  - {FIELD_DESCRIPTIONS.get(field, field)}: {val}" for field, val in filled.items())
        filled_info = f"\nInformation already collected:\n{filled_lines}\n"

    missing_list = "\n".join(f"  {i+1}. {FIELD_DESCRIPTIONS[field]}" for i, field in enumerate(missing))

    prompt = f"""You are a friendly assistant helping a marketing team member predict if one of their customers might churn.
You are talking to the MARKETING TEAM, not the customer. Ask about the customer in third person (e.g., "Does this customer have...", "What is their contract type?").
You need to ask for the following missing details about the customer.
{filled_info}
You MUST ask about ALL of these — do not skip any:
{missing_list}

RULES:
- Ask for ALL {len(missing)} items listed above. Do NOT skip any.
- Always refer to the customer in third person ("the customer", "they", "their"). NEVER say "you" or "your" as if talking to the customer.
- Be concise and friendly.
- Group related items together naturally.
- Do NOT make up or assume any values.

Write your next message to the marketing team member:"""
    return prompt


# ──────────────────────────────────────────────
# 5. SLOT EXTRACTION & VALIDATION
# ──────────────────────────────────────────────
def _message_mentions_field(user_message, field):
    """Check if the user message plausibly mentions a field (anti-hallucination guard)."""
    msg = user_message.lower()
    keywords = {
        'tenure': ['month', 'year', 'tenure', 'been with', 'been here', 'joined', 'new customer', 'long-time', 'longtime'],
        'Contract': ['contract', 'month-to-month', 'one year', 'two year', 'monthly', 'annual', 'no contract'],
        'Monthly_Charges': ['$', 'pay', 'charge', 'cost', 'bill', 'price', 'dollar', 'month', '/mo'],
        'Internet_Service': ['internet', 'fiber', 'dsl', 'broadband', 'no internet'],
        'Tech_Support': ['tech support', 'technical support', 'support'],
        'Online_Security': ['security', 'online security'],
        'Senior_Citizen': ['senior', 'elderly', 'old', 'age', 'retired'],
        'Is_Married': ['married', 'single', 'spouse', 'marital', 'wife', 'husband'],
        'Dependents': ['dependent', 'children', 'kids', 'family'],
        'Online_Backup': ['backup', 'online backup'],
        'Device_Protection': ['device', 'protection', 'device protection'],
        'Streaming_TV': ['streaming tv', 'stream tv', 'tv'],
        'Streaming_Movies': ['streaming movie', 'stream movie', 'movie'],
        'Paperless_Billing': ['paperless', 'paper', 'billing', 'e-bill'],
        'Payment_Method': ['payment', 'pay by', 'electronic check', 'credit card', 'bank transfer', 'mailed check', 'check'],
    }
    field_keywords = keywords.get(field, [])
    return any(kw in msg for kw in field_keywords)


def extract_slots_from_response(llm_response, session, user_message=""):
    """Parse LLM JSON response and validate extracted slots."""
    # Try to find JSON in the response
    try:
        # Look for JSON block in the response
        text = llm_response.strip()
        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            return {}
        json_str = text[start:end + 1]
        extracted = json.loads(json_str)
    except json.JSONDecodeError:
        return {}

    validated = {}
    for field, value in extracted.items():
        if field not in ALL_SLOTS:
            continue
        if value is None:
            continue

        # Anti-hallucination: reject if user message doesn't mention this field
        if user_message and not _message_mentions_field(user_message, field):
            continue

        # Validate and coerce
        if field in ['tenure', 'Monthly_Charges']:
            try:
                validated[field] = float(value)
            except (ValueError, TypeError):
                continue
        elif field == 'Senior_Citizen':
            try:
                val = int(value)
                if val in [0, 1]:
                    validated[field] = val
            except (ValueError, TypeError):
                continue
        elif field in VALID_VALUES:
            # Case-insensitive matching
            str_value = str(value).strip()
            for valid in VALID_VALUES[field]:
                if str_value.lower() == str(valid).lower():
                    validated[field] = valid
                    break

    return validated


# ──────────────────────────────────────────────
# 6. PREDICTION
# ──────────────────────────────────────────────
def make_prediction(session):
    """Run the ML model and return prediction results."""
    # Build input dataframe
    # Some input_columns (e.g. Total_Charges) are in the pkl but not in
    # session slots — they were filtered out because the model doesn't use them.
    # Use defaults for those.
    row = {}
    for col in input_columns:
        if col in session.slots and session.slots[col] is not None:
            row[col] = session.slots[col]
        else:
            row[col] = defaults[col]

    df = pd.DataFrame([row])

    # Ensure correct types
    for col in ['tenure', 'Monthly_Charges']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Total_Charges' in df.columns:
        df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')
    df['Senior_Citizen'] = df['Senior_Citizen'].astype(int)

    # Get prediction
    probability = pipeline.predict_proba(df)[0][1]
    prediction = "Yes" if probability >= 0.5 else "No"

    # Get per-customer SHAP values for this specific prediction
    preprocessor = pipeline.named_steps['preprocessor']
    if 'engineer' in pipeline.named_steps:
        engineered_df = pipeline.named_steps['engineer'].transform(df)
        transformed = preprocessor.transform(engineered_df)
    else:
        transformed = preprocessor.transform(df)

    shap_values = explainer.shap_values(transformed)

    # Pair each feature with its SHAP value for THIS customer
    shap_pairs = list(zip(feature_names, shap_values[0]))
    # Sort by absolute SHAP value (biggest impact first)
    shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_3 = shap_pairs[:3]

    # Build readable factor descriptions with direction
    top_3_factors = []
    for feat_name, shap_val in top_3:
        display_name = FEATURE_DISPLAY_NAMES.get(feat_name, feat_name)
        if shap_val > 0:
            direction = "increasing churn risk"
        else:
            direction = "reducing churn risk"
        top_3_factors.append({
            'name': display_name,
            'direction': direction,
            'impact': round(abs(float(shap_val)), 4),
        })

    return {
        'churn_probability': round(float(probability), 4),
        'churn_prediction': prediction,
        'top_3_risk_factors': top_3_factors,
        'mode_used': session.mode,
    }


def build_explanation(prediction_result):
    """Generate a plain English explanation of the prediction."""
    prob = prediction_result['churn_probability']
    pred = prediction_result['churn_prediction']
    factors = prediction_result['top_3_risk_factors']
    mode = prediction_result['mode_used']

    prob_pct = round(prob * 100, 1)

    # Build factor lines with direction
    factor_lines = ""
    for i, f in enumerate(factors):
        factor_lines += f"{i+1}. **{f['name']}** — {f['direction']}\n"

    if pred == "Yes":
        risk_level = "HIGH" if prob > 0.7 else "MODERATE"
        explanation = (
            f"Based on the customer data provided ({mode} analysis), this customer has a "
            f"**{risk_level} risk** of churning with a probability of **{prob_pct}%**.\n\n"
            f"The top factors influencing this prediction are:\n"
            f"{factor_lines}\n"
            f"**Recommendation:** Proactive retention measures are advised for this customer."
        )
    else:
        risk_level = "LOW" if prob < 0.3 else "MODERATE"
        explanation = (
            f"Based on the customer data provided ({mode} analysis), this customer has a "
            f"**{risk_level} risk** of churning with a probability of **{prob_pct}%**.\n\n"
            f"The top factors in this prediction are:\n"
            f"{factor_lines}\n"
            f"**Recommendation:** This customer appears stable, but continue monitoring."
        )

    return explanation


# ──────────────────────────────────────────────
# 7. MAIN CHAT HANDLER
# ──────────────────────────────────────────────
def handle_message(session, user_message):
    """
    Process a user message and return a response.
    Returns: dict with 'response' and optionally 'prediction'
    """
    if session.prediction_made:
        return {
            'response': "I've already provided a prediction for this customer. "
                        "To check a different customer, please start a new session "
                        "by clicking the **New Session** button above."
        }

    # Step 1: Extract slots from user message
    extraction_prompt = build_extraction_prompt(user_message, session)
    extraction_response = call_ollama(extraction_prompt)

    if extraction_response.startswith("ERROR:"):
        return {'response': extraction_response}

    extracted = extract_slots_from_response(extraction_response, session, user_message)

    # Step 2: Update session slots
    for field, value in extracted.items():
        session.slots[field] = value

    # Step 3: Check if we can make a prediction
    if session.all_required_filled():
        prediction = make_prediction(session)
        explanation = build_explanation(prediction)

        # In quick mode, check confidence
        if session.mode == 'quick':
            prob = prediction['churn_probability']
            if 0.2 <= prob <= 0.8:
                # Confidence not high enough — switch to full mode
                session.mode = 'full'
                missing = session.missing_slots()
                conv_prompt = build_conversation_prompt(session)
                conv_response = call_ollama(conv_prompt)

                if conv_response.startswith("ERROR:"):
                    return {'response': conv_response}

                return {
                    'response': (
                        f"Thanks! Based on the initial data, the churn probability is "
                        f"{round(prob * 100, 1)}%, which is in an uncertain range. "
                        f"I need a few more details for a more accurate prediction.\n\n"
                        f"{conv_response}"
                    )
                }

        # Prediction is confident enough (or full mode complete)
        session.prediction_made = True
        prediction['explanation'] = explanation
        return {
            'response': explanation,
            'prediction': prediction,
        }

    # Step 4: Still missing fields — ask for them
    conv_prompt = build_conversation_prompt(session)
    conv_response = call_ollama(conv_prompt)

    if conv_response.startswith("ERROR:"):
        return {'response': conv_response}

    # Acknowledge what was extracted
    if extracted:
        ack_parts = [f"{FIELD_DESCRIPTIONS.get(f, f)}: {v}" for f, v in extracted.items()]
        ack = "Got it! I've recorded: " + ", ".join(ack_parts) + ".\n\n"
        return {'response': ack + conv_response}

    # Nothing extracted — could be an out-of-scope message
    if not session.filled_slots():
        # No fields collected at all yet — first message was off-topic
        return {
            'response': "I'm here to help predict customer churn! "
                        "Please describe a customer you'd like to check — for example, "
                        "their tenure, contract type, monthly charges, and internet service. "
                        "I'll then assess their churn risk."
        }

    return {'response': conv_response}

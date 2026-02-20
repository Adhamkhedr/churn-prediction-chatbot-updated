import streamlit as st
import requests
import uuid

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Churn Prediction Chatbot", page_icon="📊", layout="centered")

# ──────────────────────────────────────────────
# CUSTOM STYLING
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    .stChatMessage { font-size: 14px; }
    .prediction-box {
        background: #14532d;
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 16px;
        margin-top: 8px;
        color: #e2e8f0;
    }
    .prediction-box.high-risk {
        background: #450a0a;
        border-color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Churn Prediction Chatbot")
with col2:
    if st.button("New Session", type="secondary"):
        # Reset API session
        try:
            requests.post(f"{API_URL}/reset", json={"session_id": st.session_state.session_id})
        except Exception:
            pass
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# ──────────────────────────────────────────────
# WELCOME MESSAGE
# ──────────────────────────────────────────────
if not st.session_state.messages:
    st.info(
        "Welcome! I help predict customer churn.\n\n"
        "Tell me about a customer and I'll assess their churn risk.\n\n"
        '**Example:** "I want to check a customer who has been with us for 3 months '
        'on a month-to-month contract, paying $70/month with fiber optic internet."'
    )

# ──────────────────────────────────────────────
# CHAT HISTORY
# ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show prediction box if present
        if "prediction" in msg:
            pred = msg["prediction"]
            prob = pred["churn_probability"]
            is_high = prob > 0.5
            risk_class = "high-risk" if is_high else ""

            factors_html = "".join(
                f"&nbsp;&nbsp;{i+1}. <strong>{f['name']}</strong> — {f['direction']}<br>"
                for i, f in enumerate(pred["top_3_risk_factors"])
            )

            st.markdown(
                f'<div class="prediction-box {risk_class}">'
                f"<strong>Prediction Result</strong><br><br>"
                f"Churn Probability: <strong>{prob * 100:.1f}%</strong><br>"
                f"Prediction: <strong>{pred['churn_prediction']}</strong><br>"
                f"Mode: <strong>{pred['mode_used']}</strong><br>"
                f"Top Risk Factors:<br>{factors_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

# ──────────────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────────────
if user_input := st.chat_input("Type your message..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    f"{API_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": user_input},
                    timeout=300,
                )
                data = res.json()
                response_text = data.get("response", "Sorry, something went wrong.")
                prediction = data.get("prediction", None)
            except requests.exceptions.ConnectionError:
                response_text = "Cannot connect to the API. Make sure the server is running (python -m uvicorn api.main:app --reload)."
                prediction = None
            except Exception as e:
                response_text = f"Error: {str(e)}"
                prediction = None

        st.markdown(response_text)

        # Build message record
        bot_msg = {"role": "assistant", "content": response_text}

        # Show prediction box if present
        if prediction:
            bot_msg["prediction"] = prediction
            prob = prediction["churn_probability"]
            is_high = prob > 0.5
            risk_class = "high-risk" if is_high else ""

            factors_html = "".join(
                f"&nbsp;&nbsp;{i+1}. <strong>{f['name']}</strong> — {f['direction']}<br>"
                for i, f in enumerate(prediction["top_3_risk_factors"])
            )

            st.markdown(
                f'<div class="prediction-box {risk_class}">'
                f"<strong>Prediction Result</strong><br><br>"
                f"Churn Probability: <strong>{prob * 100:.1f}%</strong><br>"
                f"Prediction: <strong>{prediction['churn_prediction']}</strong><br>"
                f"Mode: <strong>{prediction['mode_used']}</strong><br>"
                f"Top Risk Factors:<br>{factors_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.session_state.messages.append(bot_msg)

# AI-Powered Churn Prediction Chatbot

A Proof of Concept that combines a machine learning churn classification model with an LLM-powered chatbot. The marketing team describes a customer in plain English, and the system predicts whether that customer is likely to churn -with a probability score, risk level, and personalized risk factors.

## Project Structure

```
├── api/
│   └── main.py                  # FastAPI backend (/chat, /reset, /health endpoints)
├── chatbot/
│   └── pipeline.py              # Chat pipeline: LLM extraction, slot filling, prediction
├── model/
│   ├── preprocessing.py         # Data loading, cleaning, feature engineering
│   ├── train_model.py           # Model training, tuning, evaluation, SHAP analysis
│   ├── eda.py                   # Exploratory data analysis with plot generation
│   ├── churn_pipeline.pkl       # Trained model (ready to use)
│   └── eda_plots/               # EDA and SHAP visualizations
├── ui/
│   └── app.py                   # Streamlit chat interface
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── docs/
│   ├── project_report.md        # Technical & business report
│   ├── chatbot_user_guide.md    # User guide for the marketing team
│   └── architecture_diagram.png # System architecture and data flow
├── run.py                       # Single command to launch API + UI
└── requirements.txt             # Python dependencies
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ML Model | XGBoost | Churn classification with hyperparameter tuning |
| LLM | Mistral 7B via Ollama | Open-source, local -extracts structured data from natural language |
| API | FastAPI | REST endpoints for chat interaction |
| UI | Streamlit | Chat interface for the marketing team |
| Explainability | SHAP (TreeExplainer) | Per-customer risk factor explanations |

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed

### 1. Clone and install dependencies

```bash
git clone https://github.com/Adhamkhedr/churn-prediction-chatbot-updated.git
cd churn-prediction-chatbot-updated
pip install -r requirements.txt
```

### 2. Pull the Mistral model

```bash
ollama pull mistral
```

### 3. Run

```bash
ollama serve        # Start Ollama (skip if already running)
python run.py       # Starts both API and Streamlit UI
```

The Streamlit UI will open automatically in your browser.

### Alternative: Run components separately

```bash
# Terminal 1: Ollama
ollama serve

# Terminal 2: API server
python -m uvicorn api.main:app --reload

# Terminal 3: Streamlit UI
streamlit run ui/app.py
```

## How It Works

1. **User** describes a customer in natural language via the Streamlit chat UI
2. **Mistral LLM** (via Ollama) extracts structured fields from the message
3. **Chatbot pipeline** validates extracted data and asks follow-up questions for missing fields
4. **XGBoost model** predicts churn probability once enough data is collected
5. **SHAP TreeExplainer** identifies the top 3 factors driving this specific customer's prediction
6. **Response** is returned with probability, risk level, personalized factors, and a recommendation

### Quick Mode vs Full Mode

- **Quick Mode** collects the 6 most influential features (identified by SHAP analysis). If the prediction is confident (below 20% or above 80%), the result is returned immediately.
- **Full Mode** is triggered when the quick prediction is uncertain (20-80%). The chatbot asks for 9 additional features to improve accuracy.

## Try It Out

Copy-paste these into the chatbot to test different scenarios:

**High risk customer:**
> I have a customer who has been with us for 2 months on a month-to-month contract, paying $95/month with fiber optic internet. They have no tech support and no online security.

**Low risk customer:**
> Check a customer with 60 months tenure, two year contract, $30 monthly, DSL internet, has tech support and online security.

**Worst case profile:**
> New customer, just 1 month, month-to-month contract, $100/month, fiber optic, no tech support, no online security.

**Best case profile:**
> Long-time customer, 70 months tenure, two year contract, $25/month, DSL, has tech support and online security.

**Partial info (chatbot will ask follow-ups):**
> The customer pays $70 a month.

**Multi-turn conversation:**
> I want to check a customer on a month-to-month contract with fiber optic internet.

Then follow up with:
> They've been with us for 8 months and pay $80 a month. No tech support, no online security.

**Uncertain prediction (triggers full mode):**
> Customer with 24 months tenure, one year contract, $55 monthly charges, DSL internet, has tech support but no online security.

If it asks for more details, provide:
> They're not a senior citizen, married with no dependents. They have online backup but no device protection. They stream both TV and movies. They use paperless billing and pay by electronic check.

**Session reset:** After any prediction, click "New Session" to check a different customer.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Send message: `{"session_id": "abc", "message": "..."}` |
| `/reset` | POST | Reset session: `{"session_id": "abc"}` |

## Model Performance

| Metric | Score |
|--------|-------|
| Recall | 0.810 |
| F1 Score | 0.621 |
| Precision | 0.503 |
| ROC-AUC | 0.845 |

Optimized for **recall** -catching customers who are about to churn is more valuable than avoiding false positives.

## Retraining the Model

The trained model (`model/churn_pipeline.pkl`) is included and ready to use. To retrain:

```bash
python model/train_model.py
```

This runs hyperparameter tuning (RandomizedSearchCV, 50 iterations), evaluates against a logistic regression baseline, generates SHAP plots, and saves the best pipeline.

## Documentation

- [Project Report](docs/project_report.md) -Technical & business motivations, model explanation, client requirements alignment
- [Chatbot User Guide](docs/chatbot_user_guide.md) -How the marketing team uses the chatbot
- [Architecture Diagram](docs/architecture_diagram.png) -System architecture and data flow

## Performance Note

Mistral 7B runs locally via Ollama. On CPU, each response takes 30-60 seconds. With an NVIDIA GPU, responses are near-instant. This latency is inherent to running a local LLM and not a limitation of the pipeline implementation.

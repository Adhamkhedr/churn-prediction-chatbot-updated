from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add project root to path so we can import chatbot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from chatbot.pipeline import Session, handle_message

app = FastAPI(title="Churn Prediction Chatbot API")

# In-memory session storage
sessions: dict[str, Session] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ResetRequest(BaseModel):
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    if req.session_id not in sessions:
        sessions[req.session_id] = Session()

    session = sessions[req.session_id]
    result = handle_message(session, req.message)
    return result


@app.post("/reset")
def reset(req: ResetRequest):
    if req.session_id in sessions:
        del sessions[req.session_id]
    return {"status": "session reset", "session_id": req.session_id}

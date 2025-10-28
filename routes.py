from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(
    detials="API for asking questions about Islamic history using a RAG system."
    )

class QuestionRequest(BaseModel):
    question: str

app.get("/")
def read_root():
    return {"message": "Welcome to the Islamic History RAG API. Use the /ask endpoint to ask questions."}

app.post("/ask")
def ask_question(request: dict):
    from retrieve import ask
    question = request.get("question", "")
    if not question:
        return {"error": "Question field is required."}
    answer = ask(question)
    return {"answer": answer}


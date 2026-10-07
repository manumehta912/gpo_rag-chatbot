from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from rag import RAG

app = FastAPI(title="GPO Annual Report RAG Chatbot")
rag = RAG()


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


@app.post("/api/chat")
def chat(req: ChatRequest):
    return rag.answer(req.question, req.history)


@app.get("/")
def home():
    return FileResponse(config.BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=config.BASE_DIR / "static"), name="static")

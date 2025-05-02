from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.qa_logic import call_rag_fallback
from app.data_loader import load_data_and_index
import numpy as np

app = FastAPI()

questions, answers, index, embedding_model = load_data_and_index()

class QuestionRequest(BaseModel):
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    user_embedding = embedding_model.encode([request.question])
    distances, indices = index.search(np.array(user_embedding).astype("float32"), k=5)

    context_chunks = []
    for idx in indices[0]:
        if idx < len(questions):
            context_chunks.append(f"Q: {questions[idx]}\nA: {answers[idx]}")

    answer = call_rag_fallback(request.question, context_chunks)
    return {"answer": answer}

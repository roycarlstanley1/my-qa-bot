# === Imports ===
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss

# === App setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Data ===
# Load your CSV
df = pd.read_csv('DMQAPairs.csv')
questions = df['Question'].tolist()
answers = df['Answer'].tolist()

# Embed all questions
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(questions)

# Build FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(question_embeddings))

# === Request Body ===
class QuestionRequest(BaseModel):
    question: str

# === Endpoint ===
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    user_question = request.question

    # Embed user's question
    user_embedding = model.encode([user_question])

    # Search top 3 closest matches
    distances, indices = index.search(np.array(user_embedding).astype('float32'), k=3)

    THRESHOLD = 1.5  # you can adjust this based on real-world testing

    # If all matches are too far, fallback
    if all(dist > THRESHOLD for dist in distances[0]):
        return {"answer": "Sorry, I don't know the answer to that."}

    # Otherwise, suggest multiple options
    response = "Here's what you can try:\n"
    for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if distance <= THRESHOLD:
            response += f"{rank}. {answers[idx]}\n"

    response += "\nIf none of these help, please contact support."

    return {"answer": response}

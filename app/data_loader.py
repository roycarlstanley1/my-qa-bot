import pandas as pd
import numpy as np
import os
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

def load_data_and_index():
    # Load data
    data_path = Path(__file__).resolve().parent.parent / "data" / "DMQAPairs.csv"
    df = pd.read_csv(data_path)
    questions = df["Question"].tolist()
    answers = df["Answer"].tolist()

    # Embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embeddings = embedding_model.encode(questions)

    # FAISS Index
    dimension = question_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(question_embeddings))

    return questions, answers, index, embedding_model

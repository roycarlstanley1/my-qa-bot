import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Load CSV
df = pd.read_csv('DMQAPairs.csv')

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode questions
questions = df['Question'].tolist()
question_embeddings = model.encode(questions)

# Build FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(question_embeddings))

# Save everything
faiss.write_index(index, 'qa.index')
with open('answers.pkl', 'wb') as f:
    pickle.dump(df['Answer'].tolist(), f)

print("Data loaded and index created!")

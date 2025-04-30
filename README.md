# My QA Bot

A FastAPI-based question-answering bot that uses semantic search to match user questions with known answers from a CSV file.

## 🔧 Features

- Loads Q&A pairs from a CSV file
- Embeds questions with SentenceTransformer
- Uses FAISS for fast nearest-neighbor search
- Returns top 1–3 relevant answers
- Friendly fallback messaging
- Fully REST API-powered (via FastAPI)
- Cross-origin support for frontend use

## 🚀 How to Run

1. Create virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the server:
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

4. Visit docs:
    - Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
    - Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 📂 File Overview

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app and Q&A logic |
| `load_data.py` | (Optional) Preprocess CSV and save embeddings |
| `DMQAPairs.csv` | Your Q&A data source |
| `frontend.html` | Test page (optional) |
| `.gitignore` | Keeps repo clean |
| `README.md` | This file |

## 📝 License

MIT License – you’re free to reuse and adapt!

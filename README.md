# My QA Bot

An intelligent Q&A assistant built with FastAPI and semantic search to match user queries against a curated internal knowledge base. Designed for internal support workflows and future integration with Teams Apps.

## 🔧 Features

- Loads domain-specific Q&A pairs from an internal CSV file
- Embeds questions using SentenceTransformer for semantic similarity
- Uses FAISS for fast vector-based nearest-neighbor search
- Dynamically returns top 1–3 relevant answers in a helpdesk-style format
- Groups suggestions by category (e.g., "DCA Connectivity")
- Implements polite fallback messaging when confidence is low
- Fully REST API-powered (FastAPI), ready for Teams or frontend integration
- CORS enabled for local frontend or cross-domain clients

## 🚀 How to Run

1. Create a virtual environment:
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

4. API documentation is available at:
    - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
    - ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 📂 File Overview

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app logic and question processing |
| `load_data.py` | (Optional) Preprocess and embed Q&A data |
| `DMQAPairs.csv` | Internal Q&A data source (not included in repo) |
| `frontend.html` | Local HTML test interface (if used) |
| `.gitignore` | Excludes sensitive and build files |
| `requirements.txt` | Project dependencies |

## 🔒 Data & Usage Notice

This repository contains only the code logic.  
It is designed to run on internal, proprietary datasets which are **not included** for privacy and compliance reasons.

**Do not reuse this code with company data unless authorized.**

## 📄 License

**Code**: MIT License – The code logic is reusable and adaptable for other data sets.

**Data**: Not licensed for redistribution. Internal use only.


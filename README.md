🧠 FastAPI + GPT4All Hybrid Techbot

A GPU-accelerated technical assistant designed for Tier 3 network - database - software support, built with FastAPI, FAISS, and GPT4All.


🚀 Features
Local LLM (Meta-Llama-3-8B via GPT4All)

FAISS-powered semantic RAG (retrieval-augmented generation)

Fully offline — no OpenAI API required

Custom prompt persona

Frontend served directly via FastAPI

🗂 Project Structure

├── app/

│   ├── main.py            # FastAPI backend

│   ├── qa_logic.py        # GPT4All integration

│   ├── data_loader.py     # Loads FAISS index + Q&A pairs

│   └── __init__.py

├── static/

│   ├── frontend.html      # Chat UI

│   └── your-logo.png       # Background image

├── data/

│   └── DMQAPairs.csv      # Source data for FAISS

├── requirements.txt

└── README.md

🛠 Setup
bash

# Clone repo
git clone https://github.com/youruser/my-qa-bot.git
cd my-qa-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

🔗 Dependencies
Install system packages:

bash
sudo apt update && sudo apt install build-essential libopenblas-dev

Install Python packages:

pip install fastapi uvicorn numpy faiss-cpu sentence-transformers gpt4all

💾 Model Setup

Download the model:

From TheBloke/Llama-3-8B-Instruct-GGUF

Place .gguf file in:
~/.cache/gpt4all/

Set this path in qa_logic.py:

MODEL_PATH = os.path.expanduser("~/.cache/gpt4all/<your_model>.gguf")

🧠 Run It

source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then visit:

http://localhost:8000/

📦 Ignore These in Git

venv/
*.gguf
__pycache__/
.cache/
.gpt4all/

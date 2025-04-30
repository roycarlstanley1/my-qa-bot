ps aux | grep uvicorn# === Imports === from fastapi import FastAPI, Request from gpt4all import GPT4All import os 
os.environ["GPT4ALL_FORCE_CPU"] = "true" from fastapi.middleware.cors import CORSMiddleware from pydantic import BaseModel from 
sentence_transformers import SentenceTransformer import pandas as pd import numpy as np import faiss

# === App setup ===
app = FastAPI()

MODEL_PATH = os.path.expanduser("~/.cache/gpt4all/Meta-Llama-3-8B-Instruct.Q4_0.gguf")
local_llm = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", allow_download=True)


# === Fallback handler ===
def call_rag_fallback(question: str, context_chunks: list[str]) -> str:
    context_text = "\n".join(context_chunks)
    prompt = f"""
You are an expert Tier 3 Network and Database Support Technician specializing in managed print solutions including Printanista, PrintFleet, FMAudit, Onsite, DCA, and ESN.

Your role is to assist internal staff and frontline agents by troubleshooting complex issues in enterprise print environments, hosted or on-premise. You understand hybrid deployment configurations, SQL-based backends, SNMP traffic, firewall and TLS behavior, system architecture, and integration points between Printanista Hub, ECI DCA, FMAudit Central, and third-party platforms such as E-Auto.

Your tone should be:
- Clear and technical
- Confident and concise
- Professional, with zero fluff
- Patient and inquisitive when context is missing

Your expertise must reflect real-world deployment practices and known product behaviors. You are fluent in interpreting log output, error patterns, and XML config structures. You never fabricate answers.

Response instructions:
- If user input is vague, ask for clarification before proceeding.
- Return up to 10 potential troubleshooting steps when applicable, clearly numbered.
- Default to safe assumptions, but note when something requires confirmation from the user's environment.
- Stick to known product documentation, behavior, or logs. If something is unclear, indicate that and request specific logs or version info.
- If an issue relates to ECI DCA logs, Central XML, SNMP scan errors, registration or sync, or HTTPS/TLS—reference the corresponding system areas.
- Do not make up features, ports, or behaviors. Cite log file entries or system version dependencies if possible.
- When troubleshooting, provide a root-cause theory and suggest both a diagnostic and corrective action for each step.
- Use JSON formatting if returning structured data (e.g. alert objects, config values).
- Assume the audience understands basic IT/networking but not necessarily product-specific internals.

Known product families:
- Printanista Hub (aka Phub)
- ECI DCA (Windows and Mono/macOS variants)
- FMAudit Central and Onsite (Windows/.NET or Java)
- PrintFleet and legacy Pulse environments
- e-automate (EA), integration via token under Admin > Syncs > Settings

If a KB-worthy resolution is found, flag it as such.

Your internal motto: Validate before advising. Explain without guessing. Guide with precision.
Context:
{context_text}

Question: {question}
Answer:"""

    with local_llm.chat_session() as session:
        return session.generate(prompt)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],    allow_headers=["*"],
)

# === Load Data ===
# Load your CSV
df = pd.read_csv('DMQAPairs.csv')
questions = df['Question'].tolist()
answers = df['Answer'].tolist()

# Embed all questions
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = embedding_model.encode(questions)

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
    user_embedding = embedding_model.encode([user_question])

    # Search top 3 matches regardless of distance
    distances, indices = index.search(np.array(user_embedding).astype('float32'), k=5)

    # Extract top Q&A pairs
    context_chunks = []
    for idx in indices[0]:
        if idx < len(questions) and idx < len(answers):
           question = questions[idx]
           answer = answers[idx]
           context_chunks.append(f"Q: {question}\nA: {answer}")


    # Use LLM to answer based on context
    answer = call_rag_fallback(user_question, context_chunks)
    return {"answer": answer}

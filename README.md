# Insurance Claim Automation System

An AI-powered end-to-end insurance claim processing application that combines multimodal vision models, large language models, and retrieval-augmented generation (RAG) to automate damage assessment and claim decisions.

---

## Features

- **Multimodal Evidence Intake** — Upload photos, PDFs, or videos; the system extracts frames/pages automatically
- **Vision-Based Damage Assessment** — Analyzes damage images using LLaVA-style vision models (SmolVLM or Ollama)
- **RAG-Powered Policy Lookup** — Retrieves relevant policy terms via FAISS semantic search before every analysis
- **Structured Accident Report Analysis** — LLM extracts key facts from free-text incident descriptions
- **Automated Claim Decision** — Cross-checks damage against policy coverage and generates a recommendation
- **AI-Generated Communications** — Drafts personalized claim status emails for customers
- **Claims History** — Stores and retrieves past claim records from SQLite for context-aware decisions

---

## Architecture

The application is a 4-step Streamlit wizard:

```
Step 1: Upload Evidence
        └── preprocessing.py  (PDF pages, video frames → images)

Step 2: Enter Accident Report
        └── Free-text + policy selection

Step 3: AI Analysis
        ├── rag/retriever.py          → fetch policy context + claims history
        ├── formatter.py              → build vision prompts with policy context
        ├── inference.py              → vision model damage analysis (HuggingFace)
        ├── postprocess.py            → aggregate per-image results
        └── text_analysis/analyzer.py → Ollama LLM structured JSON analysis

Step 4: Decision & Communication
        ├── Policy cross-check + recommendation synthesis (Ollama)
        └── communication/email_generator.py → draft claim email
```

**RAG Layer (`rag/`):**
- `policy_store.py` — FAISS vector index of insurance policies (sentence-transformers embeddings); falls back to keyword search if FAISS is unavailable
- `claims_db.py` — SQLite database for claim history
- `retriever.py` — combines both sources into context injected into every LLM prompt

**Data Models (`models/schemas.py`):** Pydantic — `PolicyInfo`, `ClaimRecord`, `DamageAnalysis`, `ClaimReport`

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Vision Model | SmolVLM-256M-Instruct (HuggingFace) / Ollama |
| LLM | DeepSeek-R1:1.5b via Ollama |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Database | SQLite |
| Validation | Pydantic v2 |
| Image/PDF/Video | Pillow, PyMuPDF, OpenCV |

---

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) running locally with the DeepSeek model pulled:
  ```bash
  ollama pull deepseek-r1:1.5b
  ```
- CUDA GPU (optional but recommended for vision model inference)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/suresh2104/GenAI_application.git
cd GenAI_application

# Install dependencies
pip install -r requirements.txt
```

---

## Running the App

```bash
python run.py
# or
streamlit run main.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Configuration

All settings are controlled via environment variables (defaults shown):

| Variable | Default | Description |
|---|---|---|
| `MODEL_BACKEND` | `ollama` | `ollama` or `huggingface` |
| `OLLAMA_ENDPOINT` | `http://localhost:11434/api/generate` | Ollama server URL |
| `OLLAMA_MODEL` | `deepseek-r1:1.5b` | LLM for text analysis |
| `HF_MODEL_ID` | `HuggingFaceTB/SmolVLM-256M-Instruct` | Vision model (HuggingFace backend) |
| `VECTOR_STORE_PATH` | `./data/vector_store` | FAISS index directory |
| `CLAIMS_DB_PATH` | `./data/claims.db` | SQLite database path |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |

Set variables before running:
```bash
# Example: switch to HuggingFace backend
export MODEL_BACKEND=huggingface
python run.py
```

---

## Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_claims_db.py -v
```

Test coverage includes: claims DB CRUD, policy store retrieval, RAG retriever, email generation, prompt formatting, result aggregation, and Pydantic schema validation.

---

## Sample Data

On first run, the system automatically seeds:
- **3 insurance policies** (`INS-2024-001`, `INS-2024-002`, `INS-2024-003`) into the FAISS vector store
- **5 historical claims** into the SQLite database

These are available immediately in the sidebar without any manual setup.

---

## Project Structure

```
claim_automation/
├── main.py                      # Streamlit app entry point
├── config.py                    # Environment-based configuration
├── run.py                       # Convenience launcher
├── preprocessing.py             # File ingestion (PDF/video/image)
├── formatter.py                 # Vision model prompt builder
├── inference.py                 # HuggingFace vision inference
├── postprocess.py               # Damage result aggregation
├── rag/
│   ├── policy_store.py          # FAISS policy search
│   ├── claims_db.py             # SQLite claims history
│   └── retriever.py             # Combined RAG context
├── text_analysis/
│   └── analyzer.py              # Ollama LLM text analysis
├── communication/
│   └── email_generator.py       # Claim email generation
├── models/
│   └── schemas.py               # Pydantic data models
├── tests/                       # pytest test suite
└── data/
    ├── sample_policies.json     # Seed policy data
    ├── claims.db                # SQLite database (auto-created)
    └── vector_store/            # FAISS index (auto-created)
```

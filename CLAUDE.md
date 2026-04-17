# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the application:**
```bash
python run.py
# or
streamlit run main.py --server.port 8501
```

**Run all tests:**
```bash
pytest tests/
```

**Run a single test file:**
```bash
pytest tests/test_claims_db.py -v
```

## Architecture

This is a 4-step Streamlit wizard for AI-powered insurance claim processing. The pipeline is:

1. **Evidence Upload** → `preprocessing.py` converts PDFs (pages) and videos (frames) into image arrays
2. **Accident Report** → free-text entry with policy selection from sidebar
3. **AI Analysis** → multi-stage: RAG retrieval → vision analysis → text analysis → policy cross-check → recommendation
4. **Decision & Communication** → user override, email draft, DB save

### Key Subsystems

**RAG (`rag/`):**
- `policy_store.py` — FAISS-backed semantic search over policy documents; falls back to keyword search if FAISS is unavailable
- `claims_db.py` — SQLite CRUD for historical claims (`./data/claims.db`)
- `retriever.py` — orchestrates both, returns combined context injected into all LLM prompts

**Vision Analysis:**
- `formatter.py` — builds LLaVA-style prompts with policy context injected
- `inference.py` — runs HuggingFace SmolVLM (bfloat16, CUDA auto-detect)
- `postprocess.py` — aggregates per-image results by damage severity

**Text/LLM Analysis (`text_analysis/`, `communication/`):**
- `text_analysis/analyzer.py` — calls Ollama for structured JSON accident report analysis; strips `<think>...</think>` blocks from DeepSeek responses
- `communication/email_generator.py` — generates claim emails via Ollama; falls back to a structured template

**Data Models (`models/schemas.py`):** Pydantic models — `PolicyInfo`, `ClaimRecord`, `DamageAnalysis`, `ClaimReport`

### Configuration (`config.py`)

All settings are environment-variable-driven:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_BACKEND` | `ollama` | `ollama` or `huggingface` |
| `OLLAMA_ENDPOINT` | `http://localhost:11434/api/generate` | Ollama server |
| `OLLAMA_MODEL` | `deepseek-r1:1.5b` | LLM for text tasks |
| `HF_MODEL_ID` | `HuggingFaceTB/SmolVLM-256M-Instruct` | Vision model |
| `VECTOR_STORE_PATH` | `./data/vector_store` | FAISS index location |
| `CLAIMS_DB_PATH` | `./data/claims.db` | SQLite database |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer for embeddings |

### Graceful Degradation

The system is designed to degrade gracefully:
- FAISS unavailable → keyword search fallback
- Ollama unreachable → template-based response fallback
- All LLM calls expect JSON output; regex extraction is used to recover from malformed responses

### Session State

Streamlit session state tracks the 4-step wizard: `step`, `images`, `text_data`, `accident_report`, `selected_policy`, `vision_results`, `text_analysis`, `policy_context`, `claims_history`, `recommendation`, `email_draft`, `claim_id`, `final_decision`.

### Data Seeding

On first run, `policy_store.py` seeds 6 policy chunks into the FAISS index and `claims_db.py` pre-populates 5 sample claims. Sample policy numbers used throughout: `INS-2024-001`, `INS-2024-002`, `INS-2024-003`.

"""
Configuration settings for the Insurance Claim Automation System.
Values can be overridden via environment variables.
"""

import os

# ---------------------------------------------------------------------------
# Model backend selection
# ---------------------------------------------------------------------------
# Set MODEL_BACKEND env var to "huggingface" to use SmolVLM instead of Ollama.
MODEL_BACKEND: str = os.environ.get("MODEL_BACKEND", "ollama")

# ---------------------------------------------------------------------------
# Ollama settings
# ---------------------------------------------------------------------------
OLLAMA_ENDPOINT: str = os.environ.get(
    "OLLAMA_ENDPOINT", "http://localhost:11434/api/generate"
)
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "deepseek-r1:1.5b")

# ---------------------------------------------------------------------------
# HuggingFace settings
# ---------------------------------------------------------------------------
HF_MODEL_ID: str = os.environ.get(
    "HF_MODEL_ID", "HuggingFaceTB/SmolVLM-256M-Instruct"
)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
VECTOR_STORE_PATH: str = os.environ.get("VECTOR_STORE_PATH", "./data/vector_store")
CLAIMS_DB_PATH: str = os.environ.get("CLAIMS_DB_PATH", "./data/claims.db")

# ---------------------------------------------------------------------------
# Embedding model (sentence-transformers)
# ---------------------------------------------------------------------------
EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
APP_TITLE: str = "Insurance Claim Automation System"
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

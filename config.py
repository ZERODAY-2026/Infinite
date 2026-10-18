"""
Central configuration for ClauseLens.

All settings can be overridden via environment variables so you can adapt
the application across machines without editing application logic.
No cloud API keys exist in this file because ClauseLens runs 100% locally.
"""

import os


# ---------------------------------------------------------------------------
# Local Ollama & Gemma E2B Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv(
    "CLAUSELENS_OLLAMA_URL",
    "http://127.0.0.1:11434",
).rstrip("/")

# Run `ollama list` in your Ubuntu terminal and set CLAUSELENS_MODEL_TAG
# (or replace the default string below) with the exact model tag shown.
OLLAMA_MODEL_TAG: str = (
    os.getenv("CLAUSELENS_MODEL_TAG")
    or os.getenv("CLAUSELENS_MODEL")
    or "gemma4:e2b-it-q4_K_M"
).strip()

OLLAMA_TIMEOUT_SECONDS: int = int(os.getenv("CLAUSELENS_TIMEOUT", "120"))
OLLAMA_TEMPERATURE: float = float(os.getenv("CLAUSELENS_TEMPERATURE", "0.1"))


# ---------------------------------------------------------------------------
# Document Ingestion & Chunking Limits
# ---------------------------------------------------------------------------
MAX_PDF_SIZE_MB: int = int(os.getenv("CLAUSELENS_MAX_PDF_MB", "10"))
MAX_PDF_SIZE_BYTES: int = MAX_PDF_SIZE_MB * 1024 * 1024

CHUNK_SIZE_CHARS: int = int(os.getenv("CLAUSELENS_CHUNK_SIZE", "900"))
CHUNK_OVERLAP_CHARS: int = int(os.getenv("CLAUSELENS_CHUNK_OVERLAP", "150"))


# ---------------------------------------------------------------------------
# Local Retrieval Settings
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K: int = int(os.getenv("CLAUSELENS_TOP_K", "4"))


# ---------------------------------------------------------------------------
# Privacy & Session State Keys
# ---------------------------------------------------------------------------
ALLOWED_LOCAL_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")

SESSION_DOCUMENT_KEY: str = "clauselens_loaded_document"
SESSION_LAST_RESULT_KEY: str = "clauselens_last_result"
SESSION_LAST_QUESTION_KEY: str = "clauselens_last_question"
SESSION_UPLOAD_HASH_KEY: str = "clauselens_upload_hash"

# Keys managed in Streamlit session_state that must be purged on Clear Session
DOCUMENT_SESSION_KEYS: tuple[str, ...] = (
    SESSION_DOCUMENT_KEY,
    SESSION_LAST_RESULT_KEY,
    SESSION_LAST_QUESTION_KEY,
    SESSION_UPLOAD_HASH_KEY,
)
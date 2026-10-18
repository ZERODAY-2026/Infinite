import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:e4b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
LEGAL_SOURCES_DIR = Path(os.getenv(
    "LEGAL_SOURCES_DIR",
    "/home/govardhan/rental-legal-ai/legal_sources/india_central"
)).expanduser().resolve()
LOCAL_DATA_DIR = Path(os.getenv("LOCAL_DATA_DIR", str(PROJECT_ROOT / "local_data"))).expanduser().resolve()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
TOP_K_LEGAL = int(os.getenv("TOP_K_LEGAL", "5"))
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm", ".docx"}

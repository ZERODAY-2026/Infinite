from pathlib import Path
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import MAX_UPLOAD_MB, LEGAL_SOURCES_DIR, OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL
from .ollama_client import check_model
from .parsing import extract_bytes
from .retrieval import index_legal_sources, index_status, retrieve
from .analysis_service import analyze_document

app = FastAPI(title="Private Legal Document Triage API", version="0.1.0")
# Local-only frontend uses localhost. Do not expose this API to a public interface without security review.
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])

class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=12)

@app.get("/health")
def health():
    try:
        models = check_model()
        return {"api": "ok", "ollama": "reachable", "models": models, "index": index_status(),
                "legal_sources_dir_exists": LEGAL_SOURCES_DIR.exists()}
    except Exception as exc:
        return {"api": "ok", "ollama": "unavailable", "error": str(exc), "index": index_status(),
                "chat_model_expected": OLLAMA_CHAT_MODEL, "embedding_model_expected": OLLAMA_EMBED_MODEL}

@app.post("/sources/index")
def build_sources_index():
    try:
        return index_legal_sources()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/sources/status")
def sources_status():
    return {**index_status(), "source_directory": str(LEGAL_SOURCES_DIR)}

@app.post("/sources/search")
def search_sources(request: SearchRequest):
    try:
        results = retrieve(request.query, request.top_k)
        return {"query": request.query, "results": results, "count": len(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/documents/analyze")
async def analyze(file: UploadFile = File(...), question: str = Query(
    default="Summarise this document and identify points worth clarifying.", max_length=2000
)):
    filename = Path(file.filename or "upload.txt").name
    suffix = Path(filename).suffix.lower()
    allowed = {".pdf", ".txt", ".md", ".html", ".htm", ".docx"}
    if suffix not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Allowed: {', '.join(sorted(allowed))}")
    data = await file.read((MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit")
    try:
        pages = extract_bytes(filename, data)
        result = analyze_document(filename, pages, question)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        import logging
        logging.exception("Document analysis failed")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed ({type(exc).__name__}). Check backend terminal."
        ) from exc
    finally:
        await file.close()

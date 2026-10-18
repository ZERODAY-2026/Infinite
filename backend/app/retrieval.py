import json
import hashlib
from pathlib import Path
from typing import Any
import numpy as np

from .config import LEGAL_SOURCES_DIR, LOCAL_DATA_DIR, ALLOWED_EXTENSIONS
from .parsing import extract_file, chunk_pages
from .ollama_client import embed

INDEX_PATH = LOCAL_DATA_DIR / "legal_vectors.npz"
PASSAGES_PATH = LOCAL_DATA_DIR / "legal_passages.json"


def _source_files(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"Legal sources folder does not exist: {root}")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS and not p.name.startswith("~$"):
            yield p


def index_legal_sources(root: Path | None = None) -> dict[str, Any]:
    root = (root or LEGAL_SOURCES_DIR).resolve()
    files = list(_source_files(root))
    if not files:
        raise ValueError(f"No supported documents found in {root}. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    passages = []
    errors = []
    for path in files:
        try:
            pages = extract_file(path)
            chunks = chunk_pages(pages, max_chars=1400, overlap=180)
            for n, chunk in enumerate(chunks, start=1):
                text = chunk["text"]
                passages.append({
                    "passage_id": hashlib.sha1(f"{path}:{n}:{text[:100]}".encode()).hexdigest()[:12],
                    "source_name": path.name,
                    "source_path": str(path.relative_to(root)),
                    "page": chunk.get("page"),
                    "chunk_number": n,
                    "text": text,
                })
        except Exception as exc:
            errors.append({"file": str(path.relative_to(root)), "error": str(exc)})
    if not passages:
        raise ValueError("No readable text was extracted. Scanned PDFs may need local OCR.")

    vectors = []
    batch_size = 32
    for start in range(0, len(passages), batch_size):
        batch = passages[start:start + batch_size]
        vectors.extend(embed([p["text"] for p in batch]))
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)

    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INDEX_PATH, vectors=matrix)
    PASSAGES_PATH.write_text(json.dumps(passages, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"source_directory": str(root), "files_found": len(files), "passages_indexed": len(passages),
            "errors": errors, "index_file": str(INDEX_PATH)}


def index_status() -> dict[str, Any]:
    if not INDEX_PATH.exists() or not PASSAGES_PATH.exists():
        return {"ready": False, "passages": 0, "index_file": str(INDEX_PATH)}
    passages = json.loads(PASSAGES_PATH.read_text(encoding="utf-8"))
    return {"ready": True, "passages": len(passages), "index_file": str(INDEX_PATH)}


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    if not INDEX_PATH.exists() or not PASSAGES_PATH.exists():
        return []
    passages = json.loads(PASSAGES_PATH.read_text(encoding="utf-8"))
    vectors = np.load(INDEX_PATH)["vectors"]
    q = np.asarray(embed([query])[0], dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    scores = vectors @ q
    k = min(max(1, top_k), len(passages))
    indices = np.argsort(scores)[::-1][:k]
    return [{**passages[int(i)], "similarity": float(scores[int(i)])} for i in indices]

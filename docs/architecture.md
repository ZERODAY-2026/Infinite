# Architecture

1. Streamlit UI runs on loopback and sends uploads to FastAPI on loopback.
2. FastAPI extracts PDF page text with PyMuPDF and supports TXT/MD/HTML/DOCX. Upload data is parsed through a temporary local file which is removed when parsing completes.
3. The legal ingestion pipeline recursively reads the configured local legal-source directory, chunks extracted text while preserving PDF page references, and requests embeddings from local Ollama `embeddinggemma`.
4. Normalized vectors are saved in `local_data/legal_vectors.npz`; associated text and provenance are saved in `local_data/legal_passages.json`. This data is local and ignored by Git.
5. On analysis, document chunks and the user question are embedded locally and matched against the legal index using cosine similarity (dot product on normalized vectors).
6. Retrieved text and the document extract are sent to the configured local Ollama Gemma model to produce JSON-formatted preliminary analysis.
7. Streamlit displays summaries, findings, source passages, uncertainty, and an exportable JSON result.

No hosted LLM, OCR, embedding, storage or analytics API is intentionally used. Ollama is configured to `127.0.0.1` by default. Dependencies/model weights are acquired during setup; run an offline test and inspect dependencies before asserting no network access.

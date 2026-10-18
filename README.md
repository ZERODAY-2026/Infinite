# Private Legal Document Triage Agent (PS2 — Agentic AI: Private Legal Document Triage Agent )

A starter implementation for preliminary document understanding. It extracts a local document, retrieves semantically relevant passages from a local legal-source folder, and asks a locally running Gemma model to produce a structured summary and potential concerns. It is **not legal advice** and does not decide validity or enforceability.

## Important scope and limitations

- The default source folder is `/home/govardhan/rental-legal-ai/legal_sources/india_central` (configurable). This is read from your computer at runtime; its contents are not included in this repository/archive.
- Only for Legal Rental Documents
- Supported source/upload formats: PDF, TXT, Markdown, HTML, DOCX. Scanned PDFs need OCR; OCR is not included yet.
- The legal index is a local vector index built using `embeddinggemma`. Gemma is used meaningfully for document analysis.
- Retrieval similarity is not a probability of legal applicability. Always verify the source text, currency, jurisdiction, and applicability.
- This starter is designed for localhost use, but you must perform offline/network monitoring and code review before claiming privacy compliance. Do not expose the API or Streamlit app to the public internet.
- This is a prototype, not a production security-reviewed application. Do not use it as a substitute for qualified legal help.

## 1. Prerequisites (Ubuntu)

- Python 3.10+ (Python 3.11 recommended)
- Ollama installed and running
- Your Gemma4:e4b model installed

Check the exact local model tags:

```bash
ollama list
ollama ps
```

Pull the embedding model once during setup:

```bash
ollama pull embeddinggemma gemma4:e4b
```

Model downloads happen during setup. Normal analysis should work without internet when the model and dependencies are already present.

## 2. Configure

From the project root:

```bash
cp .env.example .env
nano .env
```

Set `OLLAMA_CHAT_MODEL` to the **exact tag** shown by `ollama list`. The default is `gemma4:e4b` because that is the tag supplied by the project owner; adjust if your tag differs. Confirm the legal source path is correct.

Load the environment for both terminals with:

```bash
set -a; source .env; set +a
```

Run this in each terminal before starting the relevant process.

## 3. Install dependencies

```bash
cd ~/legal-triage-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3 -m venv` fails on Ubuntu, install `python3-venv` with apt, then retry. Dependencies are pinned in `requirements.txt`.

## 4. Start the backend

Terminal 1:

```bash
cd ~/legal-triage-agent
source .venv/bin/activate
set -a; source .env; set +a
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health` or `http://127.0.0.1:8000/docs` in your browser. The API is bound to loopback only.

## 5. Build the local legal index

You can use the Streamlit sidebar button **Index / re-index legal sources**, or run this command in another terminal while the backend is running:

```bash
curl -X POST http://127.0.0.1:8000/sources/index
```

The backend recursively scans the configured legal-source folder for supported files, extracts text, chunks it, creates embeddings locally via Ollama, and saves the vector index under `local_data/`. This can take time for many/large files. Check status at:

```bash
curl http://127.0.0.1:8000/sources/status
```

Only index material you are permitted to process. Keep `local_data/` out of version control; it contains extracted legal text and vectors.

## 6. Start Streamlit frontend

Terminal 2:

```bash
cd ~/legal-triage-agent
source .venv/bin/activate
set -a; source .env; set +a
streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`.

## 7. Quick test

1. Confirm `/health` shows Ollama reachable and both configured models installed.
2. Index legal sources.
3. Upload a text-based rental agreement PDF or TXT.
4. Ask a focused question.
5. Inspect each finding's quoted contract passage and each retrieved legal passage. Treat model outputs as suggestions requiring verification.

You can also search the legal index directly:

```bash
curl -X POST http://127.0.0.1:8000/sources/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"termination notice period for a tenancy","top_k":5}'
```

## API endpoints

- `GET /health` — API, Ollama, model and index status
- `GET /sources/status` — local index status
- `POST /sources/index` — build/rebuild index from configured local source directory
- `POST /sources/search` — semantic search of indexed legal passages
- `POST /documents/analyze?question=...` — upload PDF/TXT/MD/HTML/DOCX and run analysis

## Privacy checklist before demo

- Run the application on the same computer, with the API bound to `127.0.0.1`.
- Download models and Python dependencies before the offline test.
- Disable networking and demonstrate a complete analysis.
- Monitor connections while indexing and analysing; inspect all code and dependencies for telemetry/remote calls.
- Inspect logs, temporary directories, Streamlit state, and `local_data/` for private content. The upload is parsed via a temporary file that is deleted after parsing; the application does not intentionally persist the uploaded agreement.
- Avoid logging prompts or uploaded text. The app currently returns a generic analysis error to avoid echoing sensitive prompts in the UI.
- Don't expose this unauthenticated prototype to a LAN/public internet.

## Known limitations

- No OCR for scanned PDFs.
- The first version uses semantic similarity only; hybrid keyword search, jurisdiction filters, source-version validation, reranking, and claim-level entailment checks should be added before relying on results.
- The analysis prompt is not a guarantee against hallucinations. Quotes and legal sources must be reviewed and tested. Citation existence does not prove legal applicability.
- Very long documents are truncated for the model; this is disclosed in metadata. The retrieval pass examines at most 30 document chunks.
- No authentication, multi-user isolation, encryption-at-rest controls, or production hardening.

## License

MIT starter code. This licence applies to this project's code only, not to any legal-source documents placed in your configured source directory. Check each source's terms independently.

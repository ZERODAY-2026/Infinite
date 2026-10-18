# Private Legal Document Triage Agent (PS2 MVP)

A local-first legal document understanding prototype for plain-text (`.txt`) files. It uses a locally installed Ollama model (default: `gemma4:e2b`) for clause interpretation. The application does not use hosted LLM APIs, cloud OCR, remote embeddings, cloud storage, analytics, or telemetry.

> **Not legal advice.** This tool supports preliminary document understanding and issue identification only. It cannot determine legal validity, enforceability, liability, or court outcomes. Consult a qualified legal professional for advice.

## MVP scope

- Accepts UTF-8 `.txt` files only (maximum 500 KB).
- Produces a plain-language overview and clause/issue triage.
- Separates explicit wording from explanation and possible concerns.
- Shows exact quotations and source line ranges; the app rejects citations not found verbatim in the uploaded text or not contained in the cited line range.
- Allows questions about the uploaded document and flags when the document is silent.
- Calls Ollama only at `http://127.0.0.1:11434`.

PDF, DOCX, OCR, vector databases, external legal research and jurisdiction-specific legal analysis are intentionally out of scope for this first text-only MVP.

## Architecture

```text
Browser (Streamlit UI)
       |
       | uploaded TXT remains in Streamlit process memory
       v
Local Python application
  - UTF-8 validation and size limit
  - line-numbered chunking
  - prompt and response validation
  - exact-quote citation verification against original text
       |
       | HTTP to loopback only: 127.0.0.1:11434
       v
Ollama local runtime -> gemma4:e2b local weights
```

No document content is intentionally sent outside the machine by this code. Ollama must be running locally. The application does not write uploaded document text to disk. The Streamlit server itself is intended to bind to localhost by default; do not expose it on a public network.

## Requirements

- Python 3.10 or later
- Ollama installed and running locally
- `gemma4:e2b` pulled into Ollama (use the exact model tag installed on your machine)
- Enough RAM/VRAM for your Gemma quantization; actual resource use depends on the model build and hardware

## Setup

```bash
# 1. Start Ollama, then acquire the model during setup:
ollama pull gemma4:e2b

# 2. Create an environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 3. Install pinned Python dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify that Ollama has the model
ollama list

# 5. Start the app (localhost only)
streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit, usually `http://127.0.0.1:8501`.

If your local Ollama model uses a different tag, change **Ollama model name** in the app sidebar. The default endpoint is `127.0.0.1:11434`; it is deliberately not configurable to an external host in this MVP.

## Tests

```bash
pytest -q
```

Tests cover line-number preservation, exact quote validation, and JSON parsing. They do not establish legal accuracy; add a human-reviewed evaluation set before deployment.

## Privacy verification (required before claiming verified offline operation)

1. Start Ollama and download Python dependencies/model weights during setup, while no private document is loaded.
2. Stop unrelated applications that may use the network.
3. Disconnect networking or run the app in a container/VM with networking disabled while keeping the local Ollama process accessible through an appropriate local setup.
4. Upload a synthetic test contract, run analysis, and ask a question.
5. Capture connection-monitoring output (for example, OS firewall/network monitor or a blocking proxy) showing no outbound connections during processing. Also inspect process connections and temporary files.
6. Review the source and dependencies for remote calls. This app's inference URL is hard-coded to loopback; Streamlit usage statistics are disabled in `.streamlit/config.toml`.

**Important:** A code review is not the same as a completed network-isolation test. Do not claim the privacy gate is proven until you have run and recorded the test on your target machine. The operating system, Ollama runtime and dependency supply chain remain part of the threat model.

## Safety and known limitations

- The model may misunderstand legal language. Always inspect the quoted original passage.
- Citation checking proves that a quote exists at the claimed line range; it does not prove the model's interpretation is correct.
- A missing finding does not prove a clause is absent or safe.
- Long documents are split into line-preserving chunks. Cross-section dependencies may be missed.
- The MVP does not establish jurisdiction, check current law, determine enforceability, or provide legal advice.
- The model may return malformed or incomplete JSON; errors are shown rather than silently replaced with fabricated output.
- Only UTF-8 plain text is supported in this version.

## Suggested next milestones

1. Add PDF text extraction locally, then scanned-PDF OCR only with local OCR.
2. Add clause navigation and source highlighting.
3. Add structured extraction for parties, dates, payment, deadlines and termination.
4. Add adversarial prompt-injection and malformed-document tests.
5. Add two-document comparison and a human-reviewed benchmark set.

## Open-source release checklist

- Choose and add an OSI-approved licence file (for example, Apache-2.0) after confirming all contributors agree.
- Keep model weights out of the repository; document the model's actual licence and acquisition steps.
- Record OS, hardware, Python, Ollama and model/quantization versions, model file size, latency and peak memory from your own test machine.
- Publish synthetic test documents only, unless redistribution rights for other documents are clear.
- Include a short demo recording showing a citation, a document-silent question, a refusal to determine enforceability, and offline/network-monitor evidence.

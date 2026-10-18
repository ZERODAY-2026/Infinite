# ClauseLens — Your Private Contract Explainer

**ClauseLens** is a local-first legal document triage prototype built for **ZeroDay OSS Hacktoberfest 2026 (Problem Statement 2: Agentic AI — Private Legal Document Triage Agent)**.

It allows a user to upload a text-based PDF agreement (such as a residential rental agreement, employment contract, or service agreement) and ask natural-language questions about its contents. Text extraction, chunking, lexical retrieval, and model inference run on your local machine using a locally installed 4-bit quantized Gemma E2B model served via Ollama.

> **Important Legal Boundary:** ClauseLens is an informational document-reading and preliminary triage tool. It is **not** a lawyer, does **not** provide legal advice, and cannot determine whether a contract clause is legally valid or enforceable.

---

## 1. Current Features
- **In-Memory PDF Ingestion:** Extracts text page by page from uploaded PDFs using `PyMuPDF` (`fitz`) while preserving 1-based page numbers without saving uploaded documents to disk.
- **Deterministic Overlapping Chunking:** Splits pages into manageable chunks with stable source identifiers (`p001_c000`, `p001_c001`, etc.).
- **Local Lexical Retrieval:** Ranks relevant chunks using an in-memory BM25-inspired / IDF-weighted lexical scorer with stopword filtering, light suffix stemming, and phrase bonuses.
- **Local Gemma E2B Inference via Ollama:** Sends only the user's question and top-$k$ retrieved chunks to `http://127.0.0.1:11434/api/generate`.
- **Citation ID Validation:** Cross-checks model-returned `source_ids` against the actual retrieved chunks, discards unknown/fabricated IDs, and displays the full text and original page number from the extracted `Chunk` object.
- **Epistemic Separation & Safe Escalation:** Distinguishes `explicit_statement`, `interpretation`, and `insufficient_evidence`, and appends a lawyer escalation notice when asked about legal validity or enforceability.
- **Per-Session State & Clearing:** Stores loaded documents in Streamlit's per-tab `st.session_state` and provides a one-click **Clear Document / Clear Session** button.

---

## 2. Explicit Limitations & Known Risks
1. **Lexical Retrieval Is Not Semantic Search:** `ClauseRetriever` relies on shared root tokens and bigrams. If a question uses vocabulary with zero lexical overlap with the contract (for example, asking about an *"automobile"* when the contract only says *"motor vehicle"*), retrieval may return zero chunks. Retrieval failure is **not** proof that a clause is absent from the document.
2. **Citation Existence vs. Factual Entailment:** `CitationValidator` proves that a cited chunk ID exists in the retrieved set and displays that chunk's authentic text and page number. It does **not** mathematically prove that the model's plain-English summary is logically entailed by the chunk. Users must always verify the answer against the displayed chunk text.
3. **Chunk-Level (Not Sentence-Level) Quotations:** Supporting passages display the full retrieved chunk (`~900` characters) rather than highlighting an individual sentence span.
4. **No OCR Support:** Scanned or image-only PDFs contain no extractable text layer and are rejected with an explicit error message.
5. **Prompt Injection Is Mitigated, Not Solved:** Untrusted contract text is wrapped in delimiter tags, system instructions forbid following embedded commands, and post-processing rejects fabricated citation IDs and obvious prompt leaks. However, small 2B models remain susceptible to sophisticated indirect prompt injection within retrieved chunks.
6. **Localhost Validation Is Not OS Network Isolation:** `src/privacy.py` blocks non-loopback Ollama URLs, and `--browser.gatherUsageStats=false` disables Streamlit's usage telemetry flag, but application code alone cannot guarantee that your operating system or browser makes zero external network connections. See `docs/privacy_design.md` for OS-level network isolation testing.

---

## 3. Architecture Summary

```text
PDF Upload (bytes in RAM)
  -> PDFReader.read_pdf_bytes()      [1-based Page objects]
  -> TextProcessor.clean_pages()     [Normalized Page objects]
  -> TextChunker.chunk_pages()       [Chunk objects with IDs like p001_c000]
  -> ClauseRetriever.retrieve()      [Top-k lexical matches]
  -> build_qa_prompt()               [Untrusted evidence block + JSON schema]
  -> GemmaClient.generate()          [POST [http://127.0.0.1:11434/api/generate](http://127.0.0.1:11434/api/generate)]
  -> parse_model_json_response()     [Strict JSON + boolean/type validation]
  -> CitationValidator               [Resolve valid IDs to original Chunk text]
  -> Streamlit UI (app.py)           [Answer, classification, chunk text, page #]
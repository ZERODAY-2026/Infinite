from __future__ import annotations
import json
import re
import html
from pathlib import Path
import requests
import streamlit as st

from triage import (
    build_chunks, analyze_chunk, answer_question, validate_citations,
    extract_json, ollama_model_available, LOCAL_OLLAMA_URL,
)

st.set_page_config(page_title="Private Legal Triage", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.28); border-radius: 16px; margin-bottom: 1rem;}
.small-muted {color: #888; font-size: .88rem;}
.evidence {border-left: 3px solid #3b82f6; padding: .55rem .85rem; background: rgba(59,130,246,.06); border-radius: 4px; margin: .45rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>⚖️ Private Legal Document Triage</h1><p>Understand a document, locate important clauses, and identify questions to raise with a qualified legal professional.</p><span class="small-muted">Local-only MVP · TXT documents · Ollama + Gemma · Not legal advice</span></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Local model")
    model = st.text_input("Ollama model name", value="gemma4:e2b")
    st.caption(f"Expected local endpoint: {LOCAL_OLLAMA_URL}")
    if st.button("Check local model", use_container_width=True):
        ok, message = ollama_model_available(model)
        (st.success if ok else st.error)(message)
    st.divider()
    st.markdown("**Privacy by design**")
    st.write("The app sends prompts only to the Ollama service on this computer (127.0.0.1). It has no cloud AI, analytics, telemetry, or document-storage integration.")
    st.warning("For a judge-ready privacy demonstration, run with networking disabled and capture connection-monitoring evidence.")

if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "document_name" not in st.session_state:
    st.session_state.document_name = ""
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "chat" not in st.session_state:
    st.session_state.chat = []

left, right = st.columns([1.05, 1.4], gap="large")
with left:
    st.subheader("1. Upload your document")
    uploaded = st.file_uploader("Choose a plain-text legal document", type=["txt"], help="Only .txt files are accepted in this MVP.")
    if uploaded is not None:
        raw = uploaded.getvalue()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error("This file is not valid UTF-8 text. Save it as UTF-8 and upload again.")
            text = ""
        if text:
            if len(text) > 500_000:
                st.error("File is larger than the MVP limit of 500 KB.")
            else:
                if st.session_state.document_name != uploaded.name or st.session_state.document_text != text:
                    st.session_state.document_text = text
                    st.session_state.document_name = uploaded.name
                    st.session_state.analysis = None
                    st.session_state.chat = []
                st.success(f"Loaded {uploaded.name} · {len(text):,} characters")
    if st.session_state.document_text:
        with st.expander("Preview source text", expanded=False):
            st.text(st.session_state.document_text[:8000] + ("\n… preview truncated" if len(st.session_state.document_text) > 8000 else ""))
        if st.button("Clear document and session", use_container_width=True):
            st.session_state.document_text = ""
            st.session_state.document_name = ""
            st.session_state.analysis = None
            st.session_state.chat = []
            st.rerun()

with right:
    st.subheader("2. Analyse the document")
    st.write("Extract key terms and concerns with source passages you can verify.")
    if not st.session_state.document_text:
        st.info("Upload a .txt document to begin.")
    else:
        st.caption(f"Current document: **{st.session_state.document_name}**")
        if st.button("Analyse document", type="primary", use_container_width=True):
            ok, msg = ollama_model_available(model)
            if not ok:
                st.error(msg)
            else:
                chunks = build_chunks(st.session_state.document_text)
                results = []
                progress = st.progress(0, text="Preparing local analysis…")
                try:
                    for i, chunk in enumerate(chunks):
                        result = analyze_chunk(model, chunk)
                        results.append(result)
                        progress.progress((i + 1) / len(chunks), text=f"Analysing section {i+1} of {len(chunks)} locally…")
                    merged = {"document_overview": "", "items": [], "limitations": []}
                    for result in results:
                        if not merged["document_overview"] and result.get("document_overview"):
                            merged["document_overview"] = result["document_overview"]
                        merged["items"].extend(result.get("items", []))
                        merged["limitations"].extend(result.get("limitations", []))
                    merged["items"] = validate_citations(merged["items"], st.session_state.document_text)
                    st.session_state.analysis = merged
                    st.session_state.chat = []
                    st.success("Analysis completed using the local model.")
                except Exception as exc:
                    st.error(f"Local analysis failed: {exc}")

if st.session_state.analysis:
    data = st.session_state.analysis
    st.divider()
    st.subheader("3. Triage report")
    st.markdown("**Document overview**")
    st.write(data.get("document_overview") or "The model did not provide an overview.")
    items = data.get("items", [])
    if not items:
        st.info("No supported clause findings were returned. This does not mean the document has no risks.")
    for idx, item in enumerate(items, 1):
        title = item.get("title") or f"Finding {idx}"
        severity = item.get("priority", "Review")
        with st.expander(f"{idx}. {title} · {severity}", expanded=(idx <= 3)):
            st.markdown("**What the document explicitly says**")
            st.write(item.get("what_text_says", "Not identified."))
            st.markdown("**Plain-language explanation**")
            st.write(item.get("plain_language", "Not available."))
            st.markdown("**Possible concern / interpretation**")
            st.write(item.get("possible_concern", "No separate concern identified."))
            st.markdown("**Why it may need attention**")
            st.write(item.get("why_review", "Not specified."))
            st.markdown("**Evidence from your document**")
            citations = item.get("citations", [])
            if citations:
                for c in citations:
                    st.markdown(f'<div class="evidence"><b>Lines {c["start_line"]}–{c["end_line"]}</b><br><code>{html.escape(c["quote"])}</code></div>', unsafe_allow_html=True)
            else:
                st.warning("No verifiable exact quotation was supplied. Treat this finding as unsupported and do not rely on it.")
            if item.get("uncertainty"):
                st.markdown("**Uncertainty**")
                st.write(item["uncertainty"])
    if data.get("limitations"):
        st.markdown("**Limitations and missing information**")
        for limitation in data["limitations"]:
            st.write(f"- {limitation}")

    st.divider()
    st.subheader("4. Ask about this document")
    st.caption("Answers are restricted to the uploaded text. If the document is silent, the assistant should say so.")
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Your question", placeholder="What does the agreement say about ending the contract?")
        submitted = st.form_submit_button("Ask locally")
    if submitted and question.strip():
        try:
            answer = answer_question(model, question.strip(), st.session_state.document_text)
            answer["items"] = validate_citations(answer.get("items", []), st.session_state.document_text)
            st.session_state.chat.append({"question": question.strip(), "answer": answer})
        except Exception as exc:
            st.error(f"Could not answer using the local model: {exc}")
    for entry in reversed(st.session_state.chat):
        st.markdown(f"**You asked:** {entry['question']}")
        a = entry["answer"]
        st.write(a.get("answer", "No answer returned."))
        if a.get("document_silent"):
            st.warning("The uploaded document does not appear to answer this question.")
        for c in a.get("citations", []):
            st.markdown(f'<div class="evidence"><b>Lines {c["start_line"]}–{c["end_line"]}</b><br><code>{html.escape(c["quote"])}</code></div>', unsafe_allow_html=True)
        if a.get("uncertainty"):
            st.caption("Uncertainty: " + a["uncertainty"])
        if a.get("escalation"):
            st.info("Consider asking a qualified legal professional: " + a["escalation"])
        st.divider()

st.caption("This tool provides preliminary document understanding and issue identification only. It is not a lawyer, does not determine legal validity, and does not provide definitive legal advice.")

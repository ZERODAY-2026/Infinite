import json
import os
import requests
import streamlit as st

API_BASE = os.getenv("LEGAL_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Private Legal Triage", page_icon="⚖️", layout="wide")
st.title("⚖️ Private Legal Document Triage")
st.caption("Local-first document understanding • Preliminary issue spotting, not legal advice")
st.info("Documents are sent only to the configured local FastAPI service. Run this app on your own computer and verify offline operation before using sensitive documents.")

with st.sidebar:
    st.subheader("Local system status")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=4).json()
        st.write("API:", health.get("api", "unknown"))
        st.write("Ollama:", health.get("ollama", "unknown"))
        st.write("Legal index:", "Ready" if health.get("index", {}).get("ready") else "Not indexed")
        if health.get("models"):
            models = health["models"]
            st.caption(f"Chat model: {models.get('chat_model')}\nEmbedding model: {models.get('embedding_model')}")
            if not models.get("chat_model_installed"):
                st.warning("Configured chat model tag was not found. Check `ollama list` and .env.")
            if not models.get("embedding_model_installed"):
                st.warning("Embedding model not found. Run `ollama pull embeddinggemma` or update .env.")
    except requests.RequestException:
        st.error("Cannot reach FastAPI. Start the backend first.")

    st.divider()
    st.subheader("Legal knowledge base")
    if st.button("Index / re-index legal sources", use_container_width=True):
        with st.spinner("Reading local legal sources and generating embeddings…"):
            try:
                response = requests.post(f"{API_BASE}/sources/index", timeout=1800)
                if response.ok:
                    st.success("Index complete")
                    st.json(response.json())
                else:
                    st.error(response.text)
            except requests.RequestException as exc:
                st.error(f"Backend request failed: {exc}")

st.subheader("Analyse a document")
uploaded = st.file_uploader("Choose a legal document", type=["pdf", "txt", "html", "docx"], help="Files are processed by your local backend. Scanned PDFs need OCR, which this starter version does not yet include.")
question = st.text_input("What would you like to understand?", value="Summarise the document and identify obligations, deadlines, termination terms, and points worth clarifying.")
if st.button("Analyse locally", type="primary", disabled=uploaded is None):
    if not question.strip():
        st.warning("Enter a question or analysis focus.")
    else:
        with st.spinner("Extracting text, retrieving local legal passages, and running Gemma…"):
            try:
                response = requests.post(
                    f"{API_BASE}/documents/analyze",
                    params={"question": question},
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                    timeout=1800,
                )
                if not response.ok:
                    st.error(f"Analysis failed: {response.text}")
                else:
                    st.session_state["analysis_result"] = response.json()
                    st.session_state["analysis_filename"] = uploaded.name
            except requests.RequestException as exc:
                st.error(f"Could not contact the local API: {exc}")

result = st.session_state.get("analysis_result")
if result:
    st.divider()
    st.header(f"Results: {st.session_state.get('analysis_filename', 'Document')}")
    st.warning("Preliminary document triage only. This output is not legal advice and does not determine legal validity or enforceability.")
    st.subheader("Plain-language summary")
    st.write(result.get("document_summary", "No summary returned."))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Important terms", "Obligations & dates", "Potential concerns", "Retrieved legal passages", "Limitations & metadata"])
    with tab1:
        terms = result.get("important_terms", [])
        if not terms:
            st.info("No important terms were returned.")
        for i, item in enumerate(terms, 1):
            with st.expander(f"{i}. {item.get('topic', 'Term')}", expanded=i == 1):
                st.write(item.get("plain_language", ""))
                if item.get("document_quote"):
                    st.markdown("**Document evidence**")
                    st.code(item["document_quote"])
                st.caption(f"Page: {item.get('page') or '-'}")
    with tab2:
        obligations = result.get("obligations_and_dates", [])
        if not obligations:
            st.info("No obligations or dates were confidently extracted.")
        for item in obligations:
            st.markdown(f"**{item.get('item', 'Item')}**")
            st.write(f"Who: {item.get('who', 'Unclear')}")
            if item.get("document_quote"):
                st.code(item["document_quote"])
            st.caption(f"Page: {item.get('page') or 'not available'}")
            st.divider()
    with tab3:
        concerns = result.get("potential_concerns", [])
        if not concerns:
            st.info("No potential concerns were returned. This does not prove the document has no risks.")
        for i, item in enumerate(concerns, 1):
            with st.expander(f"{i}. {item.get('title', 'Potential concern')} · {item.get('severity', 'review')}", expanded=i == 1):
                st.markdown("**What the document says**")
                st.write(item.get("what_document_says", ""))
                if item.get("document_quote"):
                    st.markdown("**Document evidence**")
                    st.code(item["document_quote"])
                st.markdown("**Why it may matter**")
                st.write(item.get("why_it_may_matter", ""))
                st.markdown("**Uncertainty**")
                st.write(item.get("uncertainty", ""))
                st.markdown("**Questions to clarify**")
                for q in item.get("questions_to_ask", []):
                    st.write(f"- {q}")
                for src in item.get("legal_sources", []):
                    st.markdown(f"**Retrieved source:** {src.get('source_name', 'Unknown')} · {src.get('source_path', '')} · page {src.get('page') or 'n/a'}")
                    if src.get("relevant_quote"):
                        st.code(src["relevant_quote"])
    with tab4:
        for item in result.get("retrieved_passages", []):
            with st.expander(f"{item.get('source_name')} · {item.get('source_path')} · page {item.get('page') or 'n/a'} · similarity {item.get('similarity', 0):.3f}"):
                st.write(item.get("text", ""))
                st.caption(f"Passage ID: {item.get('passage_id')}")
        if not result.get("retrieved_passages"):
            st.info("No legal passages were retrieved. Index the legal sources first.")
    with tab5:
        st.subheader("Document silence")
        for item in result.get("document_silence", []):
            st.write(f"- {item}")
        st.subheader("Limitations")
        for item in result.get("limitations", []):
            st.write(f"- {item}")
        st.write(result.get("escalation_note", ""))
        st.json(result.get("metadata", {}))
        st.caption(result.get("privacy_note", ""))

    st.download_button("Export result JSON", data=json.dumps(result, ensure_ascii=False, indent=2), file_name="legal_triage_result.json", mime="application/json")

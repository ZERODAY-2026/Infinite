"""
Streamlit user interface for ClauseLens — Your Private Contract Explainer.

Run from the clauselens/ project root:
    streamlit run app.py --browser.gatherUsageStats=false --server.address=127.0.0.1
"""

import hashlib
import streamlit as st

import config
from src.analysis import DocumentAnalysisService
from src.gemma_client import (
    GemmaClient,
    GemmaClientError,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaResponseError,
)
from src.pdf_reader import (
    PDFEmptyTextError,
    PDFMalformedError,
    PDFReadError,
    PDFSizeLimitError,
)
from src.privacy import (
    PrivacyViolationError,
    clear_document_session,
    validate_local_endpoint,
)
from src.schemas import AnalysisResult, LoadedDocument


UPLOADER_WIDGET_KEY = "clauselens_uploader_counter"


def _get_answer_type_badge(answer_type: str) -> tuple[str, str]:
    """Return a cautious, human-readable label and explanation for each answer_type."""
    if answer_type == "explicit_statement":
        return (
            "Explicit Statement (Model Classification — Verify Chunk Below)",
            "The model classified this answer as directly stated in the cited chunk(s). "
            "Always verify against the full extracted chunk text below.",
        )
    if answer_type == "interpretation":
        return (
            "Plain-English Interpretation",
            "An interpretation or summary of wording in the cited chunk(s); "
            "read the extracted chunk text carefully.",
        )
    return (
        "Insufficient Evidence / Unverified",
        "Could not be verified from the retrieved passages in the uploaded document.",
    )


def _render_sidebar() -> None:
    """Render local runtime status, privacy disclosures, and session controls."""
    with st.sidebar:
        st.header("Local Runtime & Privacy")
        st.markdown(
            f"**Ollama Endpoint:** `{config.OLLAMA_BASE_URL}`  \n"
            f"**Configured Model:** `{config.OLLAMA_MODEL_TAG}`  \n"
            f"**Max Upload Size:** `{config.MAX_PDF_SIZE_MB} MB`"
        )

        if st.button("Check Local Ollama Status", use_container_width=True):
            try:
                validate_local_endpoint(config.OLLAMA_BASE_URL)
                client = GemmaClient(
                    base_url=config.OLLAMA_BASE_URL,
                    model_tag=config.OLLAMA_MODEL_TAG,
                )
                client.verify_model_available()
                st.success(
                    f"Connected to local Ollama. Model '{config.OLLAMA_MODEL_TAG}' is available."
                )
            except (PrivacyViolationError, GemmaClientError) as exc:
                st.error(str(exc))

        st.divider()
        st.subheader("Session Controls")
        if st.button(
            "Clear Document / Clear Session",
            type="primary",
            use_container_width=True,
        ):
            clear_document_session(st.session_state)
            st.session_state[UPLOADER_WIDGET_KEY] = (
                st.session_state.get(UPLOADER_WIDGET_KEY, 0) + 1
            )
            st.rerun()

        st.divider()
        st.caption(
            "**Privacy & Legal Notice:** ClauseLens processes PDFs in memory inside "
            "your current browser session and sends prompts only to your configured "
            "local Ollama endpoint. Endpoint validation does not replace OS-level "
            "network isolation (see docs/privacy_design.md). ClauseLens provides "
            "preliminary document triage—not legal advice—and is not a substitute "
            "for a qualified lawyer."
        )


def _render_result(result: AnalysisResult) -> None:
    """Render the structured AnalysisResult including retrieved chunk text and page numbers."""
    st.subheader("Analysis Result")

    badge_label, badge_help = _get_answer_type_badge(result.answer_type)
    if result.answer_type == "explicit_statement" and result.supported_by_evidence:
        st.success(f"**Classification:** {badge_label} — {badge_help}")
    elif result.answer_type == "interpretation" and result.supported_by_evidence:
        st.warning(f"**Classification:** {badge_label} — {badge_help}")
    else:
        st.error(f"**Classification:** {badge_label} — {badge_help}")

    st.markdown("### Plain-English Answer")
    st.write(result.answer)

    st.markdown("### Uncertainty, Scope & Limitations")
    st.info(
        result.uncertainty
        or "Always verify against the original contract text and consult a qualified lawyer for legal judgment."
    )

    st.markdown("### Supporting Source Chunks from Document")
    st.caption(
        "Each passage below shows the full extracted text of the retrieved chunk "
        "matching the citation ID. A valid chunk ID confirms that the chunk exists "
        "on that page; please read the passage to verify that it supports the answer."
    )
    if not result.evidence:
        st.caption("No resolved source chunks to display for this answer.")
    else:
        for idx, ev in enumerate(result.evidence, start=1):
            with st.expander(
                f"Cited Chunk {idx} — Page {ev.page_number} (Chunk ID: {ev.chunk_id})",
                expanded=True,
            ):
                st.markdown(f"**Original Page Number:** `{ev.page_number}`")
                st.markdown(f"**Source Chunk ID:** `{ev.chunk_id}`")
                st.markdown("**Full Retrieved Chunk Text (Extracted from PDF):**")
                st.code(ev.quotation, language="text")


def main() -> None:
    st.set_page_config(
        page_title="ClauseLens — Your Private Contract Explainer",
        layout="wide",
    )

    if UPLOADER_WIDGET_KEY not in st.session_state:
        st.session_state[UPLOADER_WIDGET_KEY] = 0

    _render_sidebar()

    st.title("ClauseLens")
    st.caption(
        "Your Private Contract Explainer — On-Device Legal Document Triage Powered by Local Gemma E2B"
    )

    try:
        validate_local_endpoint(config.OLLAMA_BASE_URL)
    except PrivacyViolationError as exc:
        st.error(str(exc))
        st.stop()

    service = DocumentAnalysisService()

    st.markdown("## 1. Upload a Legal Document (Text-Based PDF)")
    uploader_key = f"pdf_uploader_{st.session_state[UPLOADER_WIDGET_KEY]}"
    uploaded_file = st.file_uploader(
        f"Choose a PDF file (max {config.MAX_PDF_SIZE_MB} MB). Remains in local memory.",
        type=["pdf"],
        key=uploader_key,
    )

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()

        if (
            st.session_state.get(config.SESSION_UPLOAD_HASH_KEY) != file_hash
            or config.SESSION_DOCUMENT_KEY not in st.session_state
        ):
            try:
                with st.spinner("Extracting text and chunking document locally..."):
                    loaded_doc = service.load_document(
                        filename=uploaded_file.name,
                        pdf_bytes=pdf_bytes,
                    )
                # Clear prior document/question state before storing the new document
                clear_document_session(st.session_state)
                st.session_state[config.SESSION_DOCUMENT_KEY] = loaded_doc
                st.session_state[config.SESSION_UPLOAD_HASH_KEY] = file_hash
            except PDFSizeLimitError as exc:
                clear_document_session(st.session_state)
                st.error(f"Upload rejected: {exc}")
            except (PDFMalformedError, PDFEmptyTextError, PDFReadError) as exc:
                clear_document_session(st.session_state)
                st.error(f"Could not process PDF: {exc}")
            except Exception:
                clear_document_session(st.session_state)
                st.error(
                    "An unexpected error occurred while reading the PDF. "
                    "Please verify that the file is a valid text-based PDF."
                )

    loaded_document: LoadedDocument | None = st.session_state.get(
        config.SESSION_DOCUMENT_KEY
    )

    if loaded_document is None:
        st.info("Upload a text-based PDF contract above to begin analysis.")
        return

    st.success(f"Loaded **{loaded_document.filename}** in current session memory.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pages Extracted", loaded_document.page_count)
    col2.metric("Searchable Chunks", len(loaded_document.chunks))
    col3.metric("File Size", f"{loaded_document.file_size_bytes / 1024:.1f} KB")

    st.markdown("## 2. Ask a Question About This Document")
    st.caption(
        "Examples: 'What is the monthly rent?', "
        "'How much notice is required to terminate the agreement?', "
        "'What is the security deposit?', or 'Is the ambiguous clause legally enforceable?'"
    )

    with st.form(key="question_form", clear_on_submit=False):
        user_question = st.text_input(
            "Your question:",
            value=st.session_state.get(config.SESSION_LAST_QUESTION_KEY, ""),
            placeholder="e.g., What is the notice period for terminating this agreement?",
        )
        submit_btn = st.form_submit_button("Analyze Question Locally")

    if submit_btn:
        if not user_question.strip():
            st.warning("Please enter a question before clicking Analyze.")
        else:
            st.session_state[config.SESSION_LAST_QUESTION_KEY] = user_question.strip()
            try:
                with st.spinner(
                    f"Retrieving passages and querying local model ({config.OLLAMA_MODEL_TAG})..."
                ):
                    result = service.answer_question(
                        document=loaded_document,
                        question=user_question.strip(),
                    )
                st.session_state[config.SESSION_LAST_RESULT_KEY] = result
            except ModelNotFoundError as exc:
                st.error(f"Model Configuration Error: {exc}")
            except OllamaConnectionError as exc:
                st.error(f"Local Ollama Connection Error: {exc}")
            except OllamaResponseError as exc:
                st.error(f"Local Ollama Response Error: {exc}")
            except PrivacyViolationError as exc:
                st.error(f"Privacy Guardrail Error: {exc}")
            except Exception:
                st.error(
                    "An unexpected error occurred during analysis. Check that Ollama "
                    "is running and that your model tag in config.py is valid."
                )

    last_result: AnalysisResult | None = st.session_state.get(
        config.SESSION_LAST_RESULT_KEY
    )
    if last_result is not None:
        st.divider()
        _render_result(last_result)


if __name__ == "__main__":
    main()
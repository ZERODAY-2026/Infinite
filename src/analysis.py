"""
Central application orchestration service for ClauseLens.

Responsibilities:
- Connect PDFReader, TextProcessor, TextChunker, ClauseRetriever,
  prompt_templates, GemmaClient, CitationValidator, and privacy checks.
- Load an uploaded PDF in memory and return a LoadedDocument.
- Answer natural-language questions grounded strictly in retrieved passages.
- Handle zero-match retrieval without claiming a clause is absent from the PDF.
- Safely parse malformed or fenced JSON from the 4-bit 2B model, requiring
  strict boolean and string types.
- Refuse to present answers as supported when valid citations are missing.
- Flag legal validity / enforceability questions or claims and recommend a lawyer.
"""

import json
import re
from typing import Any, Optional

import config
from src.chunking import TextChunker
from src.citation_validator import CitationValidator
from src.gemma_client import GemmaClient
from src.pdf_reader import PDFReader
from src.privacy import validate_local_endpoint
from src.prompt_templates import build_qa_prompt
from src.retrieval import ClauseRetriever
from src.schemas import (
    VALID_ANSWER_TYPES,
    AnalysisResult,
    LoadedDocument,
)
from src.text_processor import TextProcessor


# Keywords indicating the question or answer involves a legal validity / enforceability ruling
_LEGAL_VALIDITY_RE = re.compile(
    r"\b(enforceable|unenforceable|illegal|lawful|unlawful|legally\s+valid|"
    r"legally\s+binding|void|null\s+and\s+void|violate\s+the\s+law|court\s+uphold)\b",
    re.IGNORECASE,
)

# Markers indicating the model leaked its own system prompt due to prompt injection
_PROMPT_LEAK_RE = re.compile(
    r"(CRITICAL SECURITY AND GROUNDING RULES|UNTRUSTED_DOCUMENT_EVIDENCE|"
    r"OUTPUT FORMAT REQUIREMENT:\s*Respond ONLY with a single valid JSON)",
    re.IGNORECASE,
)

_LAWYER_ESCALATION_NOTE = (
    "ClauseLens is an informational document-reading tool, not a lawyer. "
    "Determining legal validity, enforceability, or statutory compliance requires "
    "jurisdiction-specific analysis by a qualified legal professional."
)


def parse_model_json_response(raw_text: str) -> Optional[dict[str, Any]]:
    """
    Extract and parse a JSON object from the model's raw text response.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None

    cleaned = raw_text.strip()

    if "```" in cleaned:
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fence_match:
            cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except ValueError:
        pass

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = cleaned[start_idx : end_idx + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            return None

    return None


class DocumentAnalysisService:
    """Orchestrates local document ingestion, retrieval, inference, and validation."""

    def __init__(
        self,
        pdf_reader: Optional[PDFReader] = None,
        text_processor: Optional[TextProcessor] = None,
        chunker: Optional[TextChunker] = None,
        retriever: Optional[ClauseRetriever] = None,
        gemma_client: Optional[GemmaClient] = None,
        citation_validator: Optional[CitationValidator] = None,
    ) -> None:
        self.pdf_reader = pdf_reader or PDFReader(
            max_size_bytes=config.MAX_PDF_SIZE_BYTES
        )
        self.text_processor = text_processor or TextProcessor()
        self.chunker = chunker or TextChunker(
            chunk_size=config.CHUNK_SIZE_CHARS,
            chunk_overlap=config.CHUNK_OVERLAP_CHARS,
        )
        self.retriever = retriever or ClauseRetriever(
            top_k=config.RETRIEVAL_TOP_K
        )
        self.gemma_client = gemma_client or GemmaClient(
            base_url=config.OLLAMA_BASE_URL,
            model_tag=config.OLLAMA_MODEL_TAG,
            timeout=config.OLLAMA_TIMEOUT_SECONDS,
            temperature=config.OLLAMA_TEMPERATURE,
        )
        self.citation_validator = citation_validator or CitationValidator()

    def load_document(self, filename: str, pdf_bytes: bytes) -> LoadedDocument:
        """
        Extract pages from PDF bytes, normalize text, and build overlapping chunks.
        All data stays in memory.
        """
        raw_pages = self.pdf_reader.read_pdf_bytes(pdf_bytes)
        cleaned_pages = self.text_processor.clean_pages(raw_pages)
        chunks = self.chunker.chunk_pages(cleaned_pages)

        return LoadedDocument(
            filename=filename or "uploaded_document.pdf",
            file_size_bytes=len(pdf_bytes),
            page_count=len(cleaned_pages),
            pages=cleaned_pages,
            chunks=chunks,
        )

    def answer_question(
        self,
        document: LoadedDocument,
        question: str,
    ) -> AnalysisResult:
        """
        Answer a natural-language question about a LoadedDocument using local
        retrieval, local Gemma E2B inference, and strict citation validation.
        """
        clean_question = (question or "").strip()
        if not clean_question:
            return AnalysisResult(
                question="",
                answer="Please enter a non-empty question about the uploaded document.",
                answer_type="insufficient_evidence",
                supported_by_evidence=False,
                evidence=[],
                uncertainty="No question was provided.",
                retrieved_chunk_ids=[],
            )

        validate_local_endpoint(self.gemma_client.base_url)

        retrieved_chunks = self.retriever.retrieve(
            question=clean_question,
            chunks=document.chunks,
        )
        retrieved_ids = [c.chunk_id for c in retrieved_chunks]

        if not retrieved_chunks:
            return AnalysisResult(
                question=clean_question,
                answer=(
                    "Could not verify an answer from the available document text "
                    "because no relevant passages matched your question."
                ),
                answer_type="insufficient_evidence",
                supported_by_evidence=False,
                evidence=[],
                uncertainty=(
                    "Local keyword retrieval found no matching passages for the terms "
                    "in your question. Failure to retrieve a relevant passage is NOT "
                    "proof that the topic is absent from the document—the contract may "
                    "use different terminology. Consider rephrasing your question or "
                    "reviewing the document directly."
                ),
                retrieved_chunk_ids=[],
            )

        prompt = build_qa_prompt(
            question=clean_question,
            retrieved_chunks=retrieved_chunks,
        )
        raw_model_output = self.gemma_client.generate(prompt)

        parsed = parse_model_json_response(raw_model_output)
        if parsed is None:
            return AnalysisResult(
                question=clean_question,
                answer=(
                    "The local model returned a response that could not be parsed into "
                    "the required structured verification format. To avoid presenting "
                    "unverified claims, ClauseLens has withheld the unparsed output."
                ),
                answer_type="insufficient_evidence",
                supported_by_evidence=False,
                evidence=[],
                uncertainty=(
                    "Model output was malformed JSON. You can inspect the retrieved "
                    f"passages ({', '.join(retrieved_ids)}) or try asking a simpler question."
                ),
                retrieved_chunk_ids=retrieved_ids,
            )

        # Require strict types from parsed JSON fields (Issue A & Issue F fixes)
        raw_answer_val = parsed.get("answer")
        raw_answer = raw_answer_val.strip() if isinstance(raw_answer_val, str) else ""

        raw_type_val = parsed.get("answer_type")
        raw_answer_type = (
            raw_type_val.strip().lower() if isinstance(raw_type_val, str) else ""
        )

        # Require an actual boolean True; strings like "false" or "true" evaluate to False
        raw_found_val = parsed.get("found_in_document")
        found_in_doc = raw_found_val if isinstance(raw_found_val, bool) else False

        raw_uncertainty_val = parsed.get("uncertainty")
        raw_uncertainty = (
            raw_uncertainty_val.strip()
            if isinstance(raw_uncertainty_val, str)
            else ""
        )

        raw_source_ids = parsed.get("source_ids", [])

        # Guard against prompt-injection system-prompt leakage
        if _PROMPT_LEAK_RE.search(raw_answer):
            return AnalysisResult(
                question=clean_question,
                answer=(
                    "The response was withheld because untrusted text in the document "
                    "attempted to interfere with system instructions."
                ),
                answer_type="insufficient_evidence",
                supported_by_evidence=False,
                evidence=[],
                uncertainty=(
                    "Potential prompt-injection interference detected. Inspect the "
                    "document text directly."
                ),
                retrieved_chunk_ids=retrieved_ids,
            )

        if raw_answer_type not in VALID_ANSWER_TYPES:
            raw_answer_type = (
                "interpretation" if found_in_doc else "insufficient_evidence"
            )

        validated_evidence = self.citation_validator.validate_and_resolve(
            raw_source_ids=raw_source_ids,
            retrieved_chunks=retrieved_chunks,
        )

        if not found_in_doc or raw_answer_type == "insufficient_evidence":
            final_answer = raw_answer or (
                "The retrieved passages do not contain enough evidence to answer "
                "this question."
            )
            uncertainty_notes = [
                raw_uncertainty
                or "The supplied passages do not explicitly address this question.",
                "Note: Failure to retrieve a relevant passage is NOT proof that "
                "the topic is absent from the document. The contract may use "
                "different terminology.",
            ]
            return AnalysisResult(
                question=clean_question,
                answer=final_answer,
                answer_type="insufficient_evidence",
                supported_by_evidence=False,
                evidence=[],
                uncertainty=" ".join(uncertainty_notes),
                retrieved_chunk_ids=retrieved_ids,
            )

        if not raw_answer or not validated_evidence:
            return AnalysisResult(
                question=clean_question,
                answer=(
                    "An answer could not be verified because the model did not provide "
                    "both a valid answer and a valid citation ID from the retrieved passages."
                ),
                answer_type="insufficient_evidence",
                supported_by_evidence=False,
                evidence=[],
                uncertainty=(
                    "The model generated a response without a valid source citation ID "
                    "matching the retrieved passages. Unsupported claims are withheld."
                ),
                retrieved_chunk_ids=retrieved_ids,
            )

        # Check both the user's question and the model's answer for legal validity terms
        involves_legal_validity = bool(
            _LEGAL_VALIDITY_RE.search(clean_question)
            or _LEGAL_VALIDITY_RE.search(raw_answer)
        )
        final_answer_type = raw_answer_type
        uncertainty_parts: list[str] = []

        if raw_uncertainty:
            uncertainty_parts.append(raw_uncertainty)

        if involves_legal_validity:
            if final_answer_type == "explicit_statement":
                final_answer_type = "interpretation"
            uncertainty_parts.append(_LAWYER_ESCALATION_NOTE)
        elif not uncertainty_parts:
            uncertainty_parts.append(
                "Note: Valid citation IDs confirm that the cited chunk exists on the "
                "stated page, not that the summary is guaranteed to be complete. Always "
                "verify the answer against the retrieved chunk text below and consult a "
                "qualified lawyer for legal advice."
            )

        return AnalysisResult(
            question=clean_question,
            answer=raw_answer,
            answer_type=final_answer_type,
            supported_by_evidence=True,
            evidence=validated_evidence,
            uncertainty=" ".join(uncertainty_parts),
            retrieved_chunk_ids=retrieved_ids,
        )
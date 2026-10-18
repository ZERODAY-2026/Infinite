"""
Data structures shared across all ClauseLens modules.

Keeping these dataclasses minimal, typed, and consistent guarantees that
PDF extraction, chunking, retrieval, citation validation, analysis, and the
Streamlit UI communicate through explicit contracts.
"""

from dataclasses import dataclass, field
from typing import Literal


AnswerType = Literal[
    "explicit_statement",
    "interpretation",
    "insufficient_evidence",
]

VALID_ANSWER_TYPES: frozenset[str] = frozenset(
    {
        "explicit_statement",
        "interpretation",
        "insufficient_evidence",
    }
)


@dataclass(frozen=True)
class Page:
    """Represents a single extracted page from an uploaded PDF."""

    page_number: int  # 1-based original page number from the PDF
    text: str


@dataclass(frozen=True)
class Chunk:
    """
    Represents a searchable text segment derived from a specific PDF page.

    chunk_id follows a deterministic format such as 'p001_c000' (Page 1, Chunk 0).
    """

    chunk_id: str
    page_number: int
    text: str


@dataclass(frozen=True)
class Evidence:
    """
    Verified supporting passage resolved from an original application Chunk.

    Evidence objects are always constructed by CitationValidator from stored
    Chunk objects, never from model-generated page numbers or paraphrased text.
    """

    chunk_id: str
    page_number: int
    quotation: str


@dataclass
class AnalysisResult:
    """
    Structured output returned by DocumentAnalysisService for a user question.
    """

    question: str
    answer: str
    answer_type: str
    supported_by_evidence: bool
    evidence: list[Evidence] = field(default_factory=list)
    uncertainty: str = ""
    retrieved_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class LoadedDocument:
    """
    In-memory representation of the currently uploaded PDF document.
    Stored exclusively inside the individual user's Streamlit session state.
    """

    filename: str
    file_size_bytes: int
    page_count: int
    pages: list[Page] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
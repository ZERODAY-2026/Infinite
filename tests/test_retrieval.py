"""
Unit tests for src/retrieval.py.

Verifies deterministic lexical ranking, morphological stem matching, empty
input handling, metadata preservation, and tie-breaking behavior.
"""

import pytest

from src.retrieval import ClauseRetriever, tokenize
from src.schemas import Chunk


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="p001_c000",
            page_number=1,
            text=(
                "1. RENT AND DEPOSIT: The Tenant shall pay monthly rent of Rs. 18,000 "
                "and a refundable security deposit of Rs. 50,000 prior to possession."
            ),
        ),
        Chunk(
            chunk_id="p001_c001",
            page_number=1,
            text=(
                "2. MAINTENANCE: The Tenant is responsible for minor day-to-day repairs "
                "up to Rs. 1,000. Structural and plumbing repairs belong to the Landlord."
            ),
        ),
        Chunk(
            chunk_id="p002_c000",
            page_number=2,
            text=(
                "3. TERMINATION NOTICE: Either party may terminate this Agreement "
                "by giving 30 days prior written notice to the other party."
            ),
        ),
    ]


def test_relevant_chunk_ranks_first_for_security_deposit(
    sample_chunks: list[Chunk],
) -> None:
    retriever = ClauseRetriever(top_k=3)

    results = retriever.retrieve(
        question="What is the refundable security deposit amount?",
        chunks=sample_chunks,
    )

    assert len(results) >= 1
    assert results[0].chunk_id == "p001_c000"
    assert results[0].page_number == 1
    assert "Rs. 50,000" in results[0].text


def test_morphological_inflection_matches_termination_notice(
    sample_chunks: list[Chunk],
) -> None:
    """Querying 'terminating' or 'termination' should match 'terminate' / 'TERMINATION'."""
    retriever = ClauseRetriever(top_k=2)

    results = retriever.retrieve(
        question="What is the notice period for terminating this agreement?",
        chunks=sample_chunks,
    )

    assert len(results) >= 1
    assert results[0].chunk_id == "p002_c000"
    assert results[0].page_number == 2


def test_unrelated_question_returns_empty_list(sample_chunks: list[Chunk]) -> None:
    retriever = ClauseRetriever(top_k=3)

    results = retriever.retrieve(
        question="What is the tenant's passport number and visa nationality?",
        chunks=sample_chunks,
    )

    assert results == []


def test_empty_question_or_empty_chunks_handled_safely(
    sample_chunks: list[Chunk],
) -> None:
    retriever = ClauseRetriever(top_k=3)

    assert retriever.retrieve(question="", chunks=sample_chunks) == []
    assert retriever.retrieve(question="   ", chunks=sample_chunks) == []
    assert retriever.retrieve(question="What is the rent?", chunks=[]) == []


def test_stopword_only_question_returns_empty_list(
    sample_chunks: list[Chunk],
) -> None:
    retriever = ClauseRetriever(top_k=3)

    results = retriever.retrieve(
        question="What is in this agreement document?",
        chunks=sample_chunks,
    )

    assert results == []


def test_deterministic_tie_breaking_preserves_chunk_id_order() -> None:
    tied_chunks = [
        Chunk(
            chunk_id="p002_c001",
            page_number=2,
            text="The arbitration seat shall be Bengaluru.",
        ),
        Chunk(
            chunk_id="p001_c000",
            page_number=1,
            text="The arbitration seat shall be Mumbai.",
        ),
    ]
    retriever = ClauseRetriever(top_k=2)

    scored = retriever.retrieve_with_scores(
        question="Where is the arbitration seat?",
        chunks=tied_chunks,
    )

    assert len(scored) == 2
    assert scored[0].score == scored[1].score
    # Tie-breaker sorts ascending by chunk_id
    assert [item.chunk.chunk_id for item in scored] == ["p001_c000", "p002_c001"]


def test_tokenize_removes_stopwords_and_normalizes_stems() -> None:
    tokens = tokenize("What are the tenant's main obligations for terminating?")
    assert "what" not in tokens
    assert "the" not in tokens
    assert len(tokens) >= 2
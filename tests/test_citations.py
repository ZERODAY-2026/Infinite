"""
Unit tests for src/citation_validator.py.

Note on Verification Scope:
CitationValidator verifies that a cited source_id exists within the retrieved
chunks for the current query and pulls the original text and page number from
that chunk. It does NOT verify semantic entailment of the model's prose answer.
"""

import pytest

from src.citation_validator import CitationValidator, validate_citations
from src.schemas import Chunk


@pytest.fixture
def retrieved_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="p001_c000",
            page_number=1,
            text="Clause 1: Monthly rent is Rs. 18,000 payable on the 5th.",
        ),
        Chunk(
            chunk_id="p002_c001",
            page_number=2,
            text="Clause 8: Either party may terminate with 30 days notice.",
        ),
    ]


def test_valid_source_ids_resolve_to_original_chunk_metadata(
    retrieved_chunks: list[Chunk],
) -> None:
    validator = CitationValidator()

    evidence = validator.validate_and_resolve(
        raw_source_ids=["p002_c001"],
        retrieved_chunks=retrieved_chunks,
    )

    assert len(evidence) == 1
    assert evidence[0].chunk_id == "p002_c001"
    assert evidence[0].page_number == 2
    assert evidence[0].quotation == "Clause 8: Either party may terminate with 30 days notice."


def test_unknown_or_hallucinated_source_ids_are_rejected(
    retrieved_chunks: list[Chunk],
) -> None:
    evidence = validate_citations(
        raw_source_ids=["p999_c999", "invented_id", "p001_c000"],
        retrieved_chunks=retrieved_chunks,
    )

    assert len(evidence) == 1
    assert evidence[0].chunk_id == "p001_c000"
    assert evidence[0].page_number == 1


def test_unretrieved_chunk_id_cannot_be_cited(
    retrieved_chunks: list[Chunk],
) -> None:
    """
    Even if a chunk ID like 'p003_c000' looks well-formed, it must be rejected
    if it was not in retrieved_chunks passed to the model.
    """
    evidence = validate_citations(
        raw_source_ids=["p003_c000"],
        retrieved_chunks=retrieved_chunks,
    )

    assert evidence == []


def test_duplicate_source_ids_are_deduplicated_in_order(
    retrieved_chunks: list[Chunk],
) -> None:
    evidence = validate_citations(
        raw_source_ids=["p002_c001", "p001_c000", "p002_c001", "  p001_c000  "],
        retrieved_chunks=retrieved_chunks,
    )

    assert [e.chunk_id for e in evidence] == ["p002_c001", "p001_c000"]


@pytest.mark.parametrize(
    "malformed_ids",
    [
        None,
        [],
        {},
        12345,
        True,
        [None, 42, {}, [], "   "],
    ],
)
def test_malformed_source_id_types_do_not_crash(
    malformed_ids: object,
    retrieved_chunks: list[Chunk],
) -> None:
    evidence = validate_citations(
        raw_source_ids=malformed_ids,
        retrieved_chunks=retrieved_chunks,
    )
    assert evidence == []


def test_single_string_source_id_is_normalized(
    retrieved_chunks: list[Chunk],
) -> None:
    evidence = validate_citations(
        raw_source_ids="p001_c000",
        retrieved_chunks=retrieved_chunks,
    )

    assert len(evidence) == 1
    assert evidence[0].chunk_id == "p001_c000"
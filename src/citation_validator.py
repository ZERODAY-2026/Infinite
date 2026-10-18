"""
Citation validation and evidence resolution module.

Responsibilities:
- Validate model-generated source_ids against the actual retrieved Chunk objects.
- Reject fabricated, unknown, or non-string source IDs.
- Deduplicate valid source IDs while preserving citation order.
- Construct Evidence objects strictly from the original extracted Chunk text
  and original 1-based page numbers, never from model-generated claims.

Important Epistemic Limitation:
A valid source ID proves only that the cited chunk exists in the retrieved set.
It does NOT prove that the model's prose answer is logically entailed by that chunk.
"""

from typing import Any

from src.schemas import Chunk, Evidence


class CitationValidator:
    """Validates model citation IDs and resolves them to original Chunk data."""

    def validate_and_resolve(
        self,
        raw_source_ids: Any,
        retrieved_chunks: list[Chunk],
    ) -> list[Evidence]:
        """
        Validate raw_source_ids returned by the model against retrieved_chunks.

        Args:
            raw_source_ids: The 'source_ids' field parsed from model output.
                May be a list of strings, a single string, None, or malformed data.
            retrieved_chunks: The actual Chunk objects passed to the prompt.

        Returns:
            A deduplicated list of Evidence objects built exclusively from
            matching retrieved_chunks.
        """
        if not retrieved_chunks or raw_source_ids is None:
            return []

        # Normalize raw_source_ids if the model returned a single string instead of a list
        if isinstance(raw_source_ids, str):
            candidate_ids: list[Any] = [raw_source_ids]
        elif isinstance(raw_source_ids, list):
            candidate_ids = raw_source_ids
        else:
            return []

        chunk_lookup: dict[str, Chunk] = {
            chunk.chunk_id: chunk for chunk in retrieved_chunks
        }

        resolved_evidence: list[Evidence] = []
        seen_ids: set[str] = set()

        for candidate in candidate_ids:
            if not isinstance(candidate, str):
                continue

            cleaned_id = candidate.strip()
            if not cleaned_id or cleaned_id in seen_ids:
                continue

            matched_chunk = chunk_lookup.get(cleaned_id)
            if matched_chunk is None:
                # Reject unknown or hallucinated source ID
                continue

            seen_ids.add(cleaned_id)
            resolved_evidence.append(
                Evidence(
                    chunk_id=matched_chunk.chunk_id,
                    page_number=matched_chunk.page_number,
                    quotation=matched_chunk.text,
                )
            )

        return resolved_evidence


def validate_citations(
    raw_source_ids: Any,
    retrieved_chunks: list[Chunk],
) -> list[Evidence]:
    """Convenience function wrapping CitationValidator.validate_and_resolve."""
    validator = CitationValidator()
    return validator.validate_and_resolve(
        raw_source_ids=raw_source_ids,
        retrieved_chunks=retrieved_chunks,
    )
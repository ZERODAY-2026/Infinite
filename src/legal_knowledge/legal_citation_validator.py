"""Validation of legal passage references."""

from pathlib import Path

from src.legal_knowledge.ingestion import LegalKnowledgeStore


class LegalCitationValidator:
    def __init__(self, database_path: str | Path):
        self.store = LegalKnowledgeStore(database_path)

    def validate_ids(
        self,
        passage_ids: list[str],
        verified_only: bool = True,
    ) -> dict[str, list[str]]:
        """Split citation IDs into valid and invalid references."""

        valid: list[str] = []
        invalid: list[str] = []

        for passage_id in dict.fromkeys(passage_ids):
            passage = self.store.get_by_id(passage_id)

            if passage is None:
                invalid.append(passage_id)
                continue

            if verified_only and not passage.verified:
                invalid.append(passage_id)
                continue

            valid.append(passage_id)

        return {
            "valid": valid,
            "invalid": invalid,
        }

    def validate_citations(
        self,
        passage_ids: list[str],
        verified_only: bool = True,
    ) -> bool:
        """Return True only when every supplied citation ID is valid."""

        result = self.validate_ids(
            passage_ids,
            verified_only=verified_only,
        )

        return not result["invalid"]
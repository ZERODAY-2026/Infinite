from src.legal_knowledge.ingestion import LegalKnowledgeStore
from src.legal_knowledge.legal_citation_validator import (
    LegalCitationValidator,
)
from src.legal_knowledge.models import LegalPassage


def test_validates_existing_verified_citation(tmp_path):
    database = tmp_path / "legal.sqlite3"
    store = LegalKnowledgeStore(database)

    store.upsert(
        LegalPassage(
            passage_id="IND-TEST-001",
            source_title="Test source",
            source_url="https://example.org/test",
            jurisdiction="India",
            document_type="statute",
            provision="Test provision",
            text="Test passage.",
            verified=True,
        )
    )

    validator = LegalCitationValidator(database)

    assert validator.validate_citations(["IND-TEST-001"])


def test_rejects_nonexistent_citation(tmp_path):
    validator = LegalCitationValidator(
        tmp_path / "legal.sqlite3"
    )

    result = validator.validate_ids(["DOES-NOT-EXIST"])

    assert result["valid"] == []
    assert result["invalid"] == ["DOES-NOT-EXIST"]


def test_rejects_unverified_citation_by_default(tmp_path):
    database = tmp_path / "legal.sqlite3"
    store = LegalKnowledgeStore(database)

    store.upsert(
        LegalPassage(
            passage_id="UNVERIFIED-001",
            source_title="Placeholder",
            source_url="https://example.org/placeholder",
            jurisdiction="India",
            document_type="statute",
            provision="Unverified provision",
            text="Placeholder text.",
            verified=False,
        )
    )

    validator = LegalCitationValidator(database)

    assert not validator.validate_citations(["UNVERIFIED-001"])
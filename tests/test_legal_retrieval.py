from src.legal_knowledge.ingestion import LegalKnowledgeStore
from src.legal_knowledge.legal_retrieval import LegalKnowledgeRetriever
from src.legal_knowledge.models import LegalPassage


def add_passage(store, passage_id, text, jurisdiction, verified=True):
    store.upsert(
        LegalPassage(
            passage_id=passage_id,
            source_title="Test source",
            source_url="https://example.org/test",
            jurisdiction=jurisdiction,
            document_type="statute",
            provision="Test provision",
            text=text,
            verified=verified,
        )
    )


def test_retrieves_matching_verified_passage(tmp_path):
    database = tmp_path / "legal.sqlite3"
    store = LegalKnowledgeStore(database)

    add_passage(
        store,
        "IND-001",
        "The agreement describes payment of monthly rent.",
        "India",
    )

    retriever = LegalKnowledgeRetriever(database)
    results = retriever.search(
        "monthly rent payment",
        jurisdiction="India",
    )

    assert any(p.passage_id == "IND-001" for p in results)


def test_excludes_unverified_passages_by_default(tmp_path):
    database = tmp_path / "legal.sqlite3"
    store = LegalKnowledgeStore(database)

    add_passage(
        store,
        "IND-002",
        "The agreement discusses a rental deposit.",
        "India",
        verified=False,
    )

    retriever = LegalKnowledgeRetriever(database)

    results = retriever.search("rental deposit")

    assert all(p.passage_id != "IND-002" for p in results)


def test_jurisdiction_filter(tmp_path):
    database = tmp_path / "legal.sqlite3"
    store = LegalKnowledgeStore(database)

    add_passage(
        store,
        "IND-003",
        "Rules about a rental agreement and payment.",
        "India",
    )
    add_passage(
        store,
        "OTHER-003",
        "Rules about a rental agreement and payment.",
        "Other jurisdiction",
    )

    retriever = LegalKnowledgeRetriever(database)
    results = retriever.search(
        "rental agreement payment",
        jurisdiction="India",
    )

    assert all(p.jurisdiction.lower() == "india" for p in results)
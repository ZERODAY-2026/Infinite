import json

import pytest

from src.legal_knowledge.ingestion import LegalKnowledgeStore
from src.legal_knowledge.models import LegalPassage


def make_passage(
    passage_id="TEST-001",
    verified=True,
):
    return LegalPassage(
        passage_id=passage_id,
        source_title="Test source",
        source_url="https://example.org/test",
        jurisdiction="India",
        document_type="statute",
        provision="Test provision",
        text="This is test text about a rental agreement.",
        source_version="test-version",
        verified=verified,
    )


def test_upsert_and_get_by_id(tmp_path):
    database = tmp_path / "legal.sqlite3"
    store = LegalKnowledgeStore(database)

    passage = make_passage()
    store.upsert(passage)

    result = store.get_by_id("TEST-001")

    assert result is not None
    assert result.text == passage.text
    assert result.verified is True
    assert store.count() == 1


def test_upsert_updates_existing_record(tmp_path):
    store = LegalKnowledgeStore(tmp_path / "legal.sqlite3")

    store.upsert(make_passage())

    updated = make_passage()
    updated = LegalPassage(
        **{
            **updated.to_dict(),
            "text": "Updated test passage.",
        }
    )

    store.upsert(updated)

    assert store.count() == 1
    assert store.get_by_id("TEST-001").text == "Updated test passage."


def test_import_jsonl(tmp_path):
    database = tmp_path / "legal.sqlite3"
    jsonl = tmp_path / "records.jsonl"

    record = make_passage().to_dict()
    jsonl.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    store = LegalKnowledgeStore(database)

    assert store.import_jsonl(jsonl) == 1
    assert store.count() == 1


def test_invalid_jsonl_reports_line_number(tmp_path):
    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text("{not valid json}\n", encoding="utf-8")

    store = LegalKnowledgeStore(tmp_path / "legal.sqlite3")

    with pytest.raises(ValueError, match="line 1"):
        store.import_jsonl(jsonl)
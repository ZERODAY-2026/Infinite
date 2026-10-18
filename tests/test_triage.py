import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triage import build_chunks, validate_citations, extract_json


def test_chunks_keep_original_line_numbers():
    chunks = build_chunks("alpha\nbeta\ngamma", max_chars=20)
    joined = "\n".join(chunks)
    assert "[LINE 1] alpha" in joined
    assert "[LINE 2] beta" in joined
    assert "[LINE 3] gamma" in joined


def test_citation_must_match_exact_source_and_line_range():
    source = "First clause.\nRent is due on the fifth day.\nEnd."
    items = [{"title": "Rent", "citations": [
        {"quote": "Rent is due on the fifth day.", "start_line": 2, "end_line": 2},
        {"quote": "The rent is due monthly.", "start_line": 2, "end_line": 2},
        {"quote": "Rent is due on the fifth day.", "start_line": 1, "end_line": 1},
    ]}]
    result = validate_citations(items, source)
    assert len(result) == 1
    assert len(result[0]["citations"]) == 1


def test_extract_json():
    assert extract_json('{"answer":"hello"}') == {"answer": "hello"}

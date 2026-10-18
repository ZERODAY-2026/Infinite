from backend.app.parsing import clean_text, chunk_pages


def test_clean_text():
    assert clean_text("hello   world\n\n\nnext") == "hello world\n\nnext"


def test_chunk_pages_preserves_page():
    chunks = chunk_pages([{"page": 4, "text": "a" * 3200}], max_chars=1000, overlap=100)
    assert len(chunks) > 1
    assert all(item["page"] == 4 for item in chunks)
    assert all(item["text"] for item in chunks)

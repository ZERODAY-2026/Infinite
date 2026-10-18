"""
Unit tests for src/privacy.py.

Note on Security Scope:
validate_local_endpoint verifies that the configured Ollama URL string targets
a loopback address (127.0.0.1, localhost, or ::1). It does NOT prove that the
operating system or third-party processes have no external network access.
"""

import pytest

import config
from src.privacy import (
    PrivacyViolationError,
    clear_document_session,
    validate_local_endpoint,
)


@pytest.mark.parametrize(
    "valid_url",
    [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://[::1]:11434",
        "https://127.0.0.1:11434/",
    ],
)
def test_allowed_local_endpoints_are_accepted(valid_url: str) -> None:
    assert validate_local_endpoint(valid_url) is True


@pytest.mark.parametrize(
    "rejected_url",
    [
        "",
        "   ",
        "http://example.com:11434",
        "https://api.openai.com/v1",
        "http://192.168.1.50:11434",
        "http://10.0.0.5:11434",
        "http://0.0.0.0:11434",
        "ftp://127.0.0.1:11434",
        "file:///etc/passwd",
        "http://user:password@127.0.0.1:11434",
        "not_a_url_at_all",
    ],
)
def test_non_local_or_malformed_endpoints_are_rejected(rejected_url: str) -> None:
    with pytest.raises(PrivacyViolationError):
        validate_local_endpoint(rejected_url)


def test_clear_document_session_removes_document_keys_only() -> None:
    fake_session: dict[str, object] = {
        config.SESSION_DOCUMENT_KEY: "dummy_doc_object",
        config.SESSION_LAST_RESULT_KEY: "dummy_result_object",
        config.SESSION_LAST_QUESTION_KEY: "What is the rent?",
        config.SESSION_UPLOAD_HASH_KEY: "abc123hash",
        "unrelated_ui_preference": "dark_mode",
    }

    clear_document_session(fake_session)

    assert config.SESSION_DOCUMENT_KEY not in fake_session
    assert config.SESSION_LAST_RESULT_KEY not in fake_session
    assert config.SESSION_LAST_QUESTION_KEY not in fake_session
    assert config.SESSION_UPLOAD_HASH_KEY not in fake_session
    assert fake_session.get("unrelated_ui_preference") == "dark_mode"
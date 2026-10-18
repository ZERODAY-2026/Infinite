"""
Local privacy guardrails and session-clearing utilities for ClauseLens.

Responsibilities:
- Validate that the configured Ollama endpoint points exclusively to a local
  loopback address (127.0.0.1, localhost, or ::1).
- Reject remote hostnames, public IPs, or non-HTTP schemes.
- Provide a clean helper to purge document-related keys from Streamlit's
  per-user session state.

Important Privacy Scope Note:
Validating that the Ollama URL is localhost prevents accidental configuration
of remote inference servers, but endpoint validation alone does not prove the
entire operating environment is offline. Full privacy verification requires
testing with external network access disabled or monitored (see docs/privacy_design.md).
"""

from collections.abc import MutableMapping
from typing import Any
from urllib.parse import urlparse

import config


class PrivacyViolationError(ValueError):
    """
    Raised when an endpoint configuration violates ClauseLens's local-only
    processing policy.
    """


def validate_local_endpoint(
    url: str,
    allowed_hosts: tuple[str, ...] = config.ALLOWED_LOCAL_HOSTS,
) -> bool:
    """
    Validate that the given HTTP URL resolves to an allowed loopback host.

    Args:
        url: The base URL configured for Ollama (e.g., '[http://127.0.0.1:11434](http://127.0.0.1:11434)').
        allowed_hosts: Tuple of permitted loopback hostnames.

    Returns:
        True if the URL is a valid local loopback HTTP endpoint.

    Raises:
        PrivacyViolationError: If the URL is empty, malformed, uses an invalid
            scheme, embeds credentials, or points to a non-loopback host.
    """
    if not url or not isinstance(url, str) or not url.strip():
        raise PrivacyViolationError("Ollama endpoint URL cannot be empty.")

    cleaned_url = url.strip()
    parsed = urlparse(cleaned_url)

    if parsed.scheme not in ("http", "https"):
        raise PrivacyViolationError(
            f"Invalid URL scheme '{parsed.scheme}'. Only local http/https is allowed."
        )

    if parsed.username or parsed.password:
        raise PrivacyViolationError(
            "Endpoint URL must not contain embedded credentials."
        )

    hostname = (parsed.hostname or "").lower().strip()
    normalized_allowed = {h.lower().strip() for h in allowed_hosts}

    if not hostname or hostname not in normalized_allowed:
        raise PrivacyViolationError(
            f"Privacy policy violation: Refusing to connect to non-local host "
            f"'{hostname or cleaned_url}'. ClauseLens only permits local loopback "
            f"endpoints ({', '.join(allowed_hosts)})."
        )

    return True


def clear_document_session(
    session_state: MutableMapping[str, Any],
    keys_to_clear: tuple[str, ...] = config.DOCUMENT_SESSION_KEYS,
) -> None:
    """
    Remove all uploaded document content, extracted chunks, and analysis
    results from the active Streamlit session state dictionary.
    """
    for key in keys_to_clear:
        if key in session_state:
            del session_state[key]
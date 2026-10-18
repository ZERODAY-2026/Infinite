"""
Local Ollama HTTP client for Gemma E2B inference.

Responsibilities:
- Enforce that the configured Ollama base URL is strictly local via src.privacy.
- Check whether the local Ollama server is reachable.
- Verify that the configured model tag exists in `ollama list` (/api/tags)
  without automatically pulling or substituting another model.
- Send prompts to POST /api/generate with stream=False and low temperature.
- Never log sensitive prompts, document passages, or user questions.
"""

from typing import Any
import requests

import config
from src.privacy import PrivacyViolationError, validate_local_endpoint


class GemmaClientError(Exception):
    """Base exception for local Gemma/Ollama client failures."""


class OllamaConnectionError(GemmaClientError):
    """Raised when the local Ollama server cannot be reached or times out."""


class ModelNotFoundError(GemmaClientError):
    """Raised when the configured model tag is not installed in local Ollama."""


class OllamaResponseError(GemmaClientError):
    """Raised when Ollama returns an HTTP error or a malformed payload."""


class GemmaClient:
    """HTTP client communicating exclusively with the local Ollama runtime."""

    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        model_tag: str = config.OLLAMA_MODEL_TAG,
        timeout: int = config.OLLAMA_TIMEOUT_SECONDS,
        temperature: float = config.OLLAMA_TEMPERATURE,
    ) -> None:
        # Validate privacy boundary before storing or contacting the endpoint
        validate_local_endpoint(base_url)

        self.base_url = base_url.rstrip("/")
        self.model_tag = model_tag.strip()
        self.timeout = timeout
        self.temperature = temperature

    def is_server_available(self) -> bool:
        """
        Return True if the local Ollama server responds to GET /api/tags.
        """
        validate_local_endpoint(self.base_url)
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=min(5, self.timeout),
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_installed_models(self) -> list[str]:
        """
        Query GET /api/tags and return the list of locally installed model names.

        Raises:
            OllamaConnectionError: If the Ollama server is not running or unreachable.
            OllamaResponseError: If the response status or JSON format is invalid.
        """
        validate_local_endpoint(self.base_url)
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=min(10, self.timeout),
            )
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                f"Could not connect to local Ollama server at {self.base_url}. "
                "Verify that Ollama is running (`ollama serve` or systemd service)."
            ) from exc

        if response.status_code != 200:
            raise OllamaResponseError(
                f"Ollama /api/tags returned HTTP {response.status_code}."
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise OllamaResponseError(
                "Ollama /api/tags returned invalid JSON."
            ) from exc

        models_raw = payload.get("models")
        if not isinstance(models_raw, list):
            raise OllamaResponseError(
                "Malformed response from Ollama /api/tags: missing 'models' list."
            )

        installed_names: list[str] = []
        for item in models_raw:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if isinstance(name, str) and name.strip():
                    installed_names.append(name.strip())

        return installed_names

    def verify_model_available(self) -> bool:
        """
        Verify that self.model_tag is installed in the local Ollama instance.

        Raises:
            OllamaConnectionError: If Ollama is unreachable.
            ModelNotFoundError: If self.model_tag is unconfigured or not installed.
        """
        if (
            not self.model_tag
            or self.model_tag == "REPLACE_WITH_YOUR_OLLAMA_LIST_TAG"
        ):
            raise ModelNotFoundError(
                "Model tag is not configured yet. Run `ollama list` in your terminal "
                "and set OLLAMA_MODEL_TAG in config.py (or set CLAUSELENS_MODEL_TAG)."
            )

        installed = self.list_installed_models()

        # Check exact match or implicit ':latest' tag match
        if self.model_tag in installed:
            return True
        if ":" not in self.model_tag and f"{self.model_tag}:latest" in installed:
            return True

        available_str = ", ".join(installed) if installed else "(no models installed)"
        raise ModelNotFoundError(
            f"Configured model '{self.model_tag}' was not found in local Ollama. "
            f"Installed models from `ollama list`: {available_str}. "
            "Update config.py to match your installed Gemma E2B tag."
        )

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the configured local Gemma model via POST /api/generate.

        Args:
            prompt: Complete prompt string including retrieved contract passages.

        Returns:
            Raw generated string from the model.

        Raises:
            PrivacyViolationError: If the endpoint is not localhost.
            OllamaConnectionError: On connection failure or timeout.
            ModelNotFoundError: If the configured model is not available.
            OllamaResponseError: On HTTP error or invalid JSON response.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        # Re-verify privacy and model availability before transmitting prompt
        validate_local_endpoint(self.base_url)
        self.verify_model_available()

        request_payload = {
            "model": self.model_tag,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_payload,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise OllamaConnectionError(
                f"Local Ollama inference timed out after {self.timeout} seconds. "
                "Try asking a more specific question or increasing OLLAMA_TIMEOUT_SECONDS in config.py."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                f"Failed to communicate with local Ollama server at {self.base_url}."
            ) from exc

        if response.status_code == 404:
            raise ModelNotFoundError(
                f"Ollama reported that model '{self.model_tag}' was not found (HTTP 404). "
                "Check `ollama list` and update config.py."
            )

        if response.status_code != 200:
            raise OllamaResponseError(
                f"Ollama /api/generate returned HTTP {response.status_code}."
            )

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise OllamaResponseError(
                "Ollama /api/generate returned non-JSON output."
            ) from exc

        if "error" in data and data["error"]:
            raise OllamaResponseError(f"Ollama error: {data['error']}")

        generated_text = data.get("response")
        if not isinstance(generated_text, str) or not generated_text.strip():
            raise OllamaResponseError(
                "Ollama returned an empty response string."
            )

        return generated_text.strip()
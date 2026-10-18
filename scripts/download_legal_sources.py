"""
Download explicitly configured legal-source documents for ClauseLens.

This script:
- Reads source URLs from data/legal_knowledge/sources.json.
- Downloads only sources explicitly listed in that file.
- Saves downloaded files under data/legal_knowledge/raw/.
- Records download metadata.
- Does NOT extract legal provisions or mark them as verified.
- Does NOT automatically establish whether legislation is current.

Use only sources whose access and reuse terms permit downloading.
"""

import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCES_FILE = (
    PROJECT_ROOT / "data" / "legal_knowledge" / "sources.json"
)

RAW_DIRECTORY = (
    PROJECT_ROOT / "data" / "legal_knowledge" / "raw"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "legal_knowledge"
    / "download_metadata.json"
)

USER_AGENT = "ClauseLens-LegalResearch/0.1 (source downloader)"
TIMEOUT_SECONDS = 30
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024  # 25 MiB
REQUEST_DELAY_SECONDS = 1.0

ALLOWED_SCHEMES = {"https"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/json",
    "application/xml",
    "text/plain",
    "text/html",
    "application/octet-stream",
}


def load_sources() -> list[dict]:
    """Load the configured source registry."""

    if not SOURCES_FILE.exists():
        raise FileNotFoundError(
            f"Source registry not found: {SOURCES_FILE}"
        )

    with SOURCES_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    sources = data.get("sources")

    if not isinstance(sources, list):
        raise ValueError(
            "sources.json must contain a 'sources' list."
        )

    return sources


def validate_source(source: dict) -> tuple[str, str]:
    """Validate a source entry and return its ID and URL."""

    if not isinstance(source, dict):
        raise ValueError("Each source entry must be a JSON object.")

    source_id = str(source.get("source_id", "")).strip()
    url = str(source.get("url", "")).strip()

    if not source_id:
        raise ValueError("Source is missing source_id.")

    # Keep IDs safe for use in filenames.
    safe_id = "".join(
        char for char in source_id
        if char.isalnum() or char in "-_"
    )

    if not safe_id or safe_id != source_id:
        raise ValueError(
            f"Unsafe source_id: {source_id!r}"
        )

    parsed = urllib.parse.urlparse(url)

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"{source_id}: only HTTPS source URLs are allowed."
        )

    if not parsed.hostname:
        raise ValueError(
            f"{source_id}: URL does not contain a valid hostname."
        )

    if parsed.username or parsed.password:
        raise ValueError(
            f"{source_id}: URLs containing credentials are not allowed."
        )

    if parsed.port not in (None, 443):
        raise ValueError(
            f"{source_id}: non-standard HTTPS ports are not allowed."
        )

    return source_id, url


def load_metadata() -> dict:
    """Load previous download metadata if available."""

    if not METADATA_FILE.exists():
        return {"downloads": []}

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data.get("downloads"), list):
        raise ValueError(
            "download_metadata.json has an invalid structure."
        )

    return data


def save_metadata(metadata: dict) -> None:
    """Write metadata atomically."""

    METADATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = METADATA_FILE.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")

    temporary_file.replace(METADATA_FILE)


def download_source(
    source: dict,
) -> dict:
    """Download one configured source and return its metadata."""

    source_id, url = validate_source(source)

    # The extension is determined from the response content type,
    # rather than trusting an arbitrary URL filename.
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": (
                "application/pdf, application/json, "
                "application/xml, text/plain, text/html, "
                "application/octet-stream"
            ),
        },
        method="GET",
    )

    print(f"\nDownloading {source_id}")
    print(f"URL: {url}")

    try:
        with urllib.request.urlopen(
            request,
            timeout=TIMEOUT_SECONDS,
        ) as response:
            final_url = response.geturl()
            final_parsed = urllib.parse.urlparse(final_url)

            # Do not follow redirects to non-HTTPS destinations.
            if final_parsed.scheme.lower() != "https":
                raise ValueError(
                    "Redirected URL is not HTTPS; download refused."
                )

            content_type = (
                response.headers.get_content_type().lower()
            )

            if content_type not in ALLOWED_CONTENT_TYPES:
                raise ValueError(
                    f"Unsupported content type: {content_type}"
                )

            declared_length = response.headers.get("Content-Length")

            if declared_length:
                try:
                    if int(declared_length) > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            "File exceeds the 25 MiB size limit."
                        )
                except ValueError as exc:
                    if "size limit" in str(exc):
                        raise
                    raise ValueError(
                        "Invalid Content-Length header."
                    ) from exc

            content = response.read(MAX_DOWNLOAD_BYTES + 1)

            if len(content) > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    "Downloaded file exceeds the 25 MiB size limit."
                )

    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"HTTP error {exc.code} while downloading {source_id}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Network error while downloading {source_id}: {exc.reason}"
        ) from exc

    extension_by_type = {
        "application/pdf": ".pdf",
        "application/json": ".json",
        "application/xml": ".xml",
        "text/plain": ".txt",
        "text/html": ".html",
        "application/octet-stream": ".bin",
    }

    extension = extension_by_type[content_type]

    RAW_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = RAW_DIRECTORY / f"{source_id}{extension}"

    # Do not silently overwrite an existing download.
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists. "
            "Review it before removing or replacing it."
        )

    destination.write_bytes(content)

    digest = hashlib.sha256(content).hexdigest()

    result = {
        "source_id": source_id,
        "title": source.get("title", ""),
        "original_url": url,
        "final_url": final_url,
        "downloaded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "content_type": content_type,
        "size_bytes": len(content),
        "sha256": digest,
        "local_file": str(destination.relative_to(PROJECT_ROOT)),
        "verification_status": "not_verified",
        "legal_currency_status": "not_checked",
        "reuse_terms_checked": bool(
            source.get("reuse_terms_checked", False)
        ),
    }

    print(f"Saved: {destination}")
    print(f"SHA-256: {digest}")
    print("Legal verification: NOT performed")

    return result


def main() -> int:
    """Download configured sources and save a metadata record."""

    RAW_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        sources = load_sources()
        metadata = load_metadata()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Configuration error: {exc}")
        return 1

    if not sources:
        print(
            "No sources configured. Add permitted HTTPS URLs "
            "to data/legal_knowledge/sources.json."
        )
        return 0

    existing_downloads = {
        item.get("source_id"): item
        for item in metadata["downloads"]
        if isinstance(item, dict)
    }

    successful = 0
    failed = 0

    for source in sources:
        try:
            result = download_source(source)

            # Update the record for this source without duplicating it.
            existing_downloads[result["source_id"]] = result
            metadata["downloads"] = list(
                existing_downloads.values()
            )
            save_metadata(metadata)

            successful += 1

        except (
            OSError,
            ValueError,
            RuntimeError,
            urllib.error.URLError,
        ) as exc:
            failed += 1
            print(f"ERROR: {exc}")

        time.sleep(REQUEST_DELAY_SECONDS)

    print("\nDownload summary")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Metadata: {METADATA_FILE}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
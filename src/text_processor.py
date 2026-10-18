"""
Non-destructive text cleaning and normalization for legal documents.

Responsibilities:
- Normalize irregular horizontal whitespace and line breaks.
- Remove non-printable control characters and formatting noise.
- Preserve clause numbers (e.g., 'Section 4.2(a)'), dates, currency values,
  percentages, and punctuation that carry legal meaning.
"""

import re

from src.schemas import Page


# Matches ASCII control characters except newline (\n) and tab (\t)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Matches 2 or more horizontal spaces or tabs
_HORIZONTAL_SPACE_RE = re.compile(r"[ \t\u00a0]+")

# Matches 3 or more consecutive newlines (collapses them to a double newline)
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph boundaries.

    - Converts Windows/Mac line endings (\r\n, \r) to Unix (\n).
    - Replaces non-breaking spaces and multiple spaces/tabs with a single space.
    - Strips trailing spaces on individual lines.
    - Collapses 3+ consecutive blank lines into a clean paragraph break (\n\n).
    """
    if not text:
        return ""

    # Standardize line endings
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip non-printable control characters
    cleaned = _CONTROL_CHARS_RE.sub("", cleaned)

    # Normalize horizontal whitespace line by line so we keep line breaks intact
    lines = []
    for line in cleaned.split("\n"):
        normalized_line = _HORIZONTAL_SPACE_RE.sub(" ", line).strip()
        lines.append(normalized_line)

    cleaned = "\n".join(lines)
    cleaned = _EXCESS_NEWLINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def clean_page_text(text: str) -> str:
    """
    Clean extracted page text without altering legal punctuation, section
    headings, numbers, or dates.
    """
    if not text:
        return ""

    # Replace common ligatures that occasionally appear in PDF font encodings
    ligature_map = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2013": "-",  # en-dash
        "\u2014": "-",  # em-dash
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    cleaned = text
    for src_char, target_chars in ligature_map.items():
        cleaned = cleaned.replace(src_char, target_chars)

    return normalize_whitespace(cleaned)


class TextProcessor:
    """Service class for cleaning raw extracted PDF text and Page lists."""

    def clean_text(self, text: str) -> str:
        """Clean a single text string."""
        return clean_page_text(text)

    def clean_pages(self, pages: list[Page]) -> list[Page]:
        """
        Return a new list of Page objects with cleaned text while preserving
        each page's original 1-based page_number.
        """
        cleaned_pages: list[Page] = []
        for page in pages:
            cleaned_text = self.clean_text(page.text)
            cleaned_pages.append(
                Page(page_number=page.page_number, text=cleaned_text)
            )
        return cleaned_pages
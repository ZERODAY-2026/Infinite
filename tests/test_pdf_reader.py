"""
Unit tests for src/pdf_reader.py.

Uses PyMuPDF (fitz) to construct temporary PDF byte streams in memory so tests
run deterministically without committing binary PDF files to the repository.
"""

import fitz  # PyMuPDF
import pytest

from src.pdf_reader import (
    PDFEmptyTextError,
    PDFMalformedError,
    PDFReader,
    PDFSizeLimitError,
    extract_pages_from_pdf_bytes,
)


def _make_pdf_bytes(page_texts: list[str], encrypt_password: str | None = None) -> bytes:
    """Create an in-memory PDF with the given text per page and return raw bytes."""
    doc = fitz.open()
    try:
        for text in page_texts:
            page = doc.new_page()
            if text:
                page.insert_text((50, 72), text)
        if encrypt_password:
            return doc.tobytes(
                encryption=fitz.PDF_ENCRYPT_AES_256,
                owner_pw=encrypt_password,
                user_pw=encrypt_password,
            )
        return doc.tobytes()
    finally:
        doc.close()


def test_valid_single_page_pdf_extraction() -> None:
    pdf_bytes = _make_pdf_bytes(["Clause 1: Monthly rent is Rs. 18,000."])
    reader = PDFReader(max_size_bytes=1024 * 1024)

    pages = reader.read_pdf_bytes(pdf_bytes)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Monthly rent is Rs. 18,000." in pages[0].text


def test_multi_page_pdf_preserves_one_based_page_numbers() -> None:
    pdf_bytes = _make_pdf_bytes(
        [
            "Page One: Definitions and Parties.",
            "Page Two: Payment and Security Deposit of Rs. 50,000.",
            "Page Three: Termination Requires 30 Days Written Notice.",
        ]
    )
    pages = extract_pages_from_pdf_bytes(pdf_bytes)

    assert len(pages) == 3
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "Definitions and Parties" in pages[0].text
    assert "Rs. 50,000" in pages[1].text
    assert "30 Days Written Notice" in pages[2].text


def test_sparse_multi_page_pdf_preserves_page_numbers() -> None:
    """A blank first page followed by a text page must preserve page_number=2."""
    pdf_bytes = _make_pdf_bytes(["", "Clause on Page 2."])
    reader = PDFReader()

    pages = reader.read_pdf_bytes(pdf_bytes)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].text.strip() == ""
    assert pages[1].page_number == 2
    assert "Clause on Page 2." in pages[1].text


def test_empty_or_whitespace_only_pdf_raises_empty_text_error() -> None:
    blank_pdf_bytes = _make_pdf_bytes(["", "   \n\t  "])
    reader = PDFReader()

    with pytest.raises(PDFEmptyTextError, match="No extractable text"):
        reader.read_pdf_bytes(blank_pdf_bytes)


def test_malformed_pdf_bytes_raises_malformed_error() -> None:
    reader = PDFReader()

    with pytest.raises(PDFMalformedError, match="Could not read the PDF"):
        reader.read_pdf_bytes(b"This is definitely not a valid PDF header or stream.")


def test_empty_byte_input_raises_malformed_error() -> None:
    reader = PDFReader()

    with pytest.raises(PDFMalformedError, match="empty or invalid"):
        reader.read_pdf_bytes(b"")


def test_oversized_pdf_raises_size_limit_error() -> None:
    pdf_bytes = _make_pdf_bytes(["Some valid contract text."])
    # Set limit smaller than the generated PDF byte length
    strict_reader = PDFReader(max_size_bytes=50)

    with pytest.raises(PDFSizeLimitError, match="exceeds the maximum allowed limit"):
        strict_reader.read_pdf_bytes(pdf_bytes)


def test_encrypted_pdf_raises_malformed_error() -> None:
    encrypted_bytes = _make_pdf_bytes(
        ["Confidential contract clause."],
        encrypt_password="secret",
    )
    reader = PDFReader()

    with pytest.raises(PDFMalformedError, match="password-protected or encrypted"):
        reader.read_pdf_bytes(encrypted_bytes)


def test_invalid_max_size_configuration_raises_value_error() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PDFReader(max_size_bytes=0)
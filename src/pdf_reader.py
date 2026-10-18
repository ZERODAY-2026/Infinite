"""
In-memory PDF text extraction module using PyMuPDF (fitz).

Responsibilities:
- Validate uploaded PDF byte size before parsing.
- Open PDF directly from memory bytes without writing private files to disk.
- Extract text page by page while preserving 1-based page numbers.
- Detect malformed/corrupt PDFs and image-only (non-extractable) PDFs cleanly.
"""

import fitz  # PyMuPDF

import config
from src.schemas import Page


class PDFReadError(Exception):
    """Base exception for all PDF ingestion failures."""


class PDFSizeLimitError(PDFReadError):
    """Raised when the uploaded PDF exceeds the configured maximum byte size."""


class PDFMalformedError(PDFReadError):
    """Raised when the uploaded bytes are not a valid or readable PDF file."""


class PDFEmptyTextError(PDFReadError):
    """
    Raised when a PDF contains zero pages or has no extractable text
    (for example, a scanned image-only PDF when OCR is not enabled).
    """


class PDFReader:
    """Reads PDF bytes in memory and returns structured Page objects."""

    def __init__(self, max_size_bytes: int = config.MAX_PDF_SIZE_BYTES) -> None:
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be a positive integer.")
        self.max_size_bytes = max_size_bytes

    def read_pdf_bytes(self, pdf_bytes: bytes) -> list[Page]:
        """
        Extract text page by page from raw PDF bytes.

        Args:
            pdf_bytes: Raw binary content of the uploaded PDF.

        Returns:
            A list of Page objects containing 1-based page numbers and raw text.

        Raises:
            PDFSizeLimitError: If len(pdf_bytes) > self.max_size_bytes.
            PDFMalformedError: If PyMuPDF cannot open or parse the byte stream.
            PDFEmptyTextError: If the PDF is empty or contains no extractable text.
        """
        if not isinstance(pdf_bytes, (bytes, bytearray)) or len(pdf_bytes) == 0:
            raise PDFMalformedError("The uploaded file is empty or invalid.")

        if len(pdf_bytes) > self.max_size_bytes:
            max_mb = self.max_size_bytes / (1024 * 1024)
            actual_mb = len(pdf_bytes) / (1024 * 1024)
            raise PDFSizeLimitError(
                f"Uploaded PDF size ({actual_mb:.2f} MB) exceeds the maximum "
                f"allowed limit of {max_mb:.2f} MB."
            )

        try:
            doc = fitz.open(stream=bytes(pdf_bytes), filetype="pdf")
        except Exception as exc:
            raise PDFMalformedError(
                "Could not read the PDF file. Ensure the file is a valid, "
                "uncorrupted PDF document."
            ) from exc

        try:
            if doc.is_encrypted:
                raise PDFMalformedError(
                    "The uploaded PDF is password-protected or encrypted. "
                    "Please upload an unencrypted text-based PDF."
                )

            if doc.page_count == 0:
                raise PDFEmptyTextError("The uploaded PDF contains zero pages.")

            pages: list[Page] = []
            has_any_text = False

            for page_index in range(doc.page_count):
                page_obj = doc.load_page(page_index)
                raw_text = page_obj.get_text("text") or ""
                if raw_text.strip():
                    has_any_text = True
                # Preserve 1-based original page numbers even for sparse pages
                pages.append(Page(page_number=page_index + 1, text=raw_text))

            if not has_any_text:
                raise PDFEmptyTextError(
                    "No extractable text was found in this PDF. It appears to be a "
                    "scanned or image-only document. OCR is not enabled in this MVP; "
                    "please upload a text-based PDF."
                )

            return pages
        finally:
            doc.close()


def extract_pages_from_pdf_bytes(
    pdf_bytes: bytes,
    max_size_bytes: int = config.MAX_PDF_SIZE_BYTES,
) -> list[Page]:
    """Convenience function wrapping PDFReader.read_pdf_bytes."""
    reader = PDFReader(max_size_bytes=max_size_bytes)
    return reader.read_pdf_bytes(pdf_bytes)
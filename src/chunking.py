"""
Page-aware overlapping text chunker for legal documents.

Responsibilities:
- Split cleaned page text into manageable chunks with configurable size and overlap.
- Assign deterministic, human-readable chunk IDs ('p001_c000').
- Preserve original 1-based page numbers.
- Prefer natural break points (paragraph breaks, sentence endings, semicolons)
  so legal clauses are not sliced mid-word or mid-sentence when avoidable.
- Guarantee forward progress on every iteration to prevent infinite loops.
"""

import config
from src.schemas import Chunk, Page


class TextChunker:
    """Splits Page objects into overlapping Chunk objects with stable IDs."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE_CHARS,
        chunk_overlap: int = config.CHUNK_OVERLAP_CHARS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list[Page]) -> list[Chunk]:
        """
        Divide a list of Page objects into overlapping Chunk objects.

        Empty or whitespace-only pages are safely skipped without breaking
        the original page numbering of subsequent pages.
        """
        all_chunks: list[Chunk] = []

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            page_Text_chunks = self._split_single_page_text(page_text)
            for chunk_idx, segment in enumerate(page_Text_chunks):
                chunk_id = f"p{page.page_number:03d}_c{chunk_idx:03d}"
                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        page_number=page.page_number,
                        text=segment,
                    )
                )

        return all_chunks

    def _split_single_page_text(self, text: str) -> list[str]:
        """
        Split a single page's text into overlapping segments of up to chunk_size.
        """
        text_length = len(text)
        if text_length <= self.chunk_size:
            return [text]

        segments: list[str] = []
        start = 0

        while start < text_length:
            target_end = min(start + self.chunk_size, text_length)

            if target_end < text_length:
                actual_end = self._find_natural_breakpoint(text, start, target_end)
            else:
                actual_end = text_length

            segment = text[start:actual_end].strip()
            if segment:
                segments.append(segment)

            if actual_end >= text_length:
                break

            # Calculate next start index using overlap, ensuring strictly positive step
            next_start = actual_end - self.chunk_overlap
            if next_start <= start:
                next_start = start + max(1, self.chunk_size - self.chunk_overlap)

            # Advance past leading whitespace so the next chunk starts cleanly
            while next_start < text_length and text[next_start].isspace():
                next_start += 1

            start = next_start

        return segments

    def _find_natural_breakpoint(self, text: str, start: int, target_end: int) -> int:
        """
        Look backward from target_end toward the midpoint of the current window
        to find a clean clause or sentence boundary.
        """
        min_acceptable_end = start + int(self.chunk_size * 0.55)
        window = text[min_acceptable_end:target_end]

        # Priority order of boundary delimiters in legal prose
        separators = ("\n\n", ".\n", ";\n", ". ", "; ", "\n", ", ", " ")
        for sep in separators:
            pos = window.rfind(sep)
            if pos != -1:
                # Include the punctuation mark itself in the current chunk
                return min_acceptable_end + pos + len(sep.rstrip())

        return target_end


def chunk_pages(
    pages: list[Page],
    chunk_size: int = config.CHUNK_SIZE_CHARS,
    chunk_overlap: int = config.CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """Convenience function wrapping TextChunker.chunk_pages."""
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_pages(pages)
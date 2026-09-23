from dataclasses import dataclass
from typing import Any
from uuid import uuid4


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    character_count: int


def normalize_text(text: str) -> str:
    """
    Normalize extracted PDF text.

    Removes excessive whitespace while preserving paragraph boundaries.
    """

    lines = []

    for line in text.splitlines():
        cleaned = " ".join(line.split())

        if cleaned:
            lines.append(cleaned)

    return "\n\n".join(lines)


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split normalized text into paragraphs.
    """

    normalized = normalize_text(text)

    if not normalized:
        return []

    return [
        paragraph.strip()
        for paragraph in normalized.split("\n\n")
        if paragraph.strip()
    ]


def build_chunks(
    document_id: str,
    pages: list[dict[str, Any]],
    max_chunk_size: int = 1200,
    overlap_size: int = 200,
) -> list[DocumentChunk]:
    """
    Build paragraph-aware chunks while preserving page metadata.

    Args:
        document_id: Unique document identifier.
        pages: Page-level extracted PDF data.
        max_chunk_size: Maximum approximate characters per chunk.
        overlap_size: Number of characters to carry between chunks.

    Returns:
        List of DocumentChunk objects.
    """

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than 0.")

    if overlap_size < 0:
        raise ValueError("overlap_size cannot be negative.")

    if overlap_size >= max_chunk_size:
        raise ValueError(
            "overlap_size must be smaller than max_chunk_size."
        )

    chunks: list[DocumentChunk] = []

    chunk_index = 0

    for page in pages:
        page_number = int(page["page_number"])
        page_text = str(page.get("text", ""))

        paragraphs = split_into_paragraphs(page_text)

        if not paragraphs:
            continue

        current_paragraphs: list[str] = []
        current_length = 0

        for paragraph in paragraphs:
            paragraph_length = len(paragraph)

            if (
                current_paragraphs
                and current_length + paragraph_length + 2
                > max_chunk_size
            ):
                chunk_text = "\n\n".join(current_paragraphs).strip()

                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid4()),
                        document_id=document_id,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        character_count=len(chunk_text),
                    )
                )

                chunk_index += 1

                overlap_text = chunk_text[-overlap_size:]

                current_paragraphs = (
                    [overlap_text] if overlap_text else []
                )

                current_length = len(overlap_text)

            current_paragraphs.append(paragraph)
            current_length += paragraph_length + 2

        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs).strip()

            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    character_count=len(chunk_text),
                )
            )

            chunk_index += 1

    return chunks

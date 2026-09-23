from pathlib import Path
from typing import Any

from pypdf import PdfReader


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed."""


def extract_pdf_text(file_path: Path) -> dict[str, Any]:
    """
    Extract text from a PDF while preserving page-level information.

    Returns:
        {
            "page_count": int,
            "pages": [
                {
                    "page_number": int,
                    "text": str,
                    "character_count": int
                }
            ],
            "total_character_count": int
        }
    """

    if not file_path.exists():
        raise PDFProcessingError(f"PDF file not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise PDFProcessingError("Only PDF files are supported.")

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        raise PDFProcessingError(
            f"Unable to read PDF: {exc}"
        ) from exc

    pages: list[dict[str, Any]] = []
    total_character_count = 0

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise PDFProcessingError(
                f"Failed to extract text from page {page_number}: {exc}"
            ) from exc

        text = text.strip()

        page_data = {
            "page_number": page_number,
            "text": text,
            "character_count": len(text),
        }

        pages.append(page_data)
        total_character_count += len(text)

    return {
        "page_count": len(reader.pages),
        "pages": pages,
        "total_character_count": total_character_count,
    }

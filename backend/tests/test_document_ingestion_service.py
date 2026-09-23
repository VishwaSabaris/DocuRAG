from pathlib import Path
from uuid import UUID

from app.db.database import pool
from app.services.ingestion.document_ingestion_service import (
    DocumentIngestionService,
)


class FakeEmbeddingService:
    """
    Fake embedding service for testing the ingestion pipeline
    without loading the real embedding model.
    """

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [float(index + 1)] + [0.0] * 383
            for index in range(len(texts))
        ]


def create_test_pdf(path: Path) -> None:
    """
    Create a minimal PDF fixture.

    The ingestion test expects a real PDF file, so this helper
    writes a tiny PDF containing extractable text.
    """

    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n"
        b"<< /Type /Catalog /Pages 2 0 R >>\n"
        b"endobj\n"
        b"2 0 obj\n"
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
        b"endobj\n"
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> "
        b"/Contents 5 0 R >>\n"
        b"endobj\n"
        b"4 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        b"endobj\n"
        b"5 0 obj\n"
        b"<< /Length 72 >>\n"
        b"stream\n"
        b"BT\n"
        b"/F1 12 Tf\n"
        b"72 720 Td\n"
        b"(DocuRAG ingestion test document.) Tj\n"
        b"0 -20 Td\n"
        b"(This document tests PDF ingestion.) Tj\n"
        b"ET\n"
        b"endstream\n"
        b"endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000274 00000 n \n"
        b"0000000344 00000 n \n"
        b"trailer\n"
        b"<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n"
        b"465\n"
        b"%%EOF\n"
    )

    path.write_bytes(pdf_content)


def test_document_ingestion_service(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "ingestion-test.pdf"

    create_test_pdf(pdf_path)

    with pool.connection() as connection:
        service = DocumentIngestionService(
            connection=connection,
            embedding_service=FakeEmbeddingService(),
        )

        result = service.ingest_pdf(
            file_path=pdf_path,
            max_chunk_size=1200,
            overlap_size=200,
        )

        document = result["document"]

        assert document is not None

        document_id = document["id"]

        assert isinstance(document_id, UUID)

        assert document["filename"] == (
            "ingestion-test.pdf"
        )

        assert document["status"] == "processed"

        assert result["page_count"] == 1

        assert result["total_character_count"] > 0

        assert result["chunk_count"] > 0

        assert result["embedding_count"] == (
            result["chunk_count"]
        )

        repository = service.repository

        stored_chunks = (
            repository.get_document_chunks(
                document_id
            )
        )

        assert len(stored_chunks) == (
            result["chunk_count"]
        )

        for chunk in stored_chunks:
            assert chunk["document_id"] == document_id
            assert chunk["text"]
            assert chunk["embedding"] is not None


def test_document_ingestion_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_file = (
        tmp_path / "does-not-exist.pdf"
    )

    with pool.connection() as connection:
        service = DocumentIngestionService(
            connection=connection,
            embedding_service=FakeEmbeddingService(),
        )

        try:
            service.ingest_pdf(
                file_path=missing_file,
            )

            assert False, "Expected PDFProcessingError"

        except Exception as exc:
            assert "PDF file not found" in str(exc)

from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from psycopg import Connection

from app.repositories.document_repository import DocumentRepository
from app.services.chunking.chunk_service import build_chunks
from app.services.embeddings.embedding_service import (
    EmbeddingService,
    get_embedding_service,
)
from app.services.ingestion.pdf_service import (
    PDFProcessingError,
    extract_pdf_text,
)


class DocumentIngestionService:
    """
    Complete document ingestion pipeline.

    Pipeline:

        PDF
         ↓
        text extraction
         ↓
        chunking
         ↓
        embeddings
         ↓
        PostgreSQL + pgvector
    """

    def __init__(
        self,
        connection: Connection,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.repository = DocumentRepository(connection)

        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else get_embedding_service()
        )

    def ingest_pdf(
        self,
        file_path: Path,
        document_id: UUID | None = None,
        max_chunk_size: int = 1200,
        overlap_size: int = 200,
    ) -> dict[str, Any]:
        if not file_path.exists():
            raise PDFProcessingError(
                f"PDF file not found: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise PDFProcessingError(
                "Only PDF files are supported."
            )

        document_id = document_id or uuid4()
        filename = file_path.name
        stored_filename = file_path.name

        document = self.repository.create_document(
            document_id=document_id,
            filename=filename,
            stored_filename=stored_filename,
            status="processing",
        )

        try:
            extracted_data = extract_pdf_text(file_path)

            chunks = build_chunks(
                document_id=str(document_id),
                pages=extracted_data["pages"],
                max_chunk_size=max_chunk_size,
                overlap_size=overlap_size,
            )

            if not chunks:
                raise PDFProcessingError(
                    "No text chunks could be extracted from the PDF."
                )

            chunk_records = [
                {
                    "id": UUID(chunk.chunk_id),
                    "document_id": document_id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "character_count": chunk.character_count,
                }
                for chunk in chunks
            ]

            saved_chunk_count = self.repository.save_chunks(
                chunk_records
            )

            texts = [
                chunk.text
                for chunk in chunks
            ]

            embeddings = (
                self.embedding_service.embed_documents(
                    texts
                )
            )

            embedding_records = [
                {
                    "chunk_id": chunk_records[index]["id"],
                    "embedding": embeddings[index],
                }
                for index in range(len(embeddings))
            ]

            saved_embedding_count = (
                self.repository.save_chunk_embeddings(
                    embedding_records
                )
            )

            completed_document = (
                self.repository.update_document_status(
                    document_id=document_id,
                    status="processed",
                )
            )

            return {
                "document": completed_document,
                "page_count": extracted_data["page_count"],
                "total_character_count": (
                    extracted_data[
                        "total_character_count"
                    ]
                ),
                "chunk_count": saved_chunk_count,
                "embedding_count": (
                    saved_embedding_count
                ),
            }

        except Exception:
            self.repository.update_document_status(
                document_id=document_id,
                status="failed",
            )
            raise

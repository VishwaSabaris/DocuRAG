from typing import Any
from uuid import UUID

from psycopg import Connection

from app.core.config import get_settings
from app.repositories.document_repository import DocumentRepository
from app.services.embeddings.embedding_service import (
    EmbeddingService,
    get_embedding_service,
)


class RetrievalService:
    """
    Service responsible for semantic retrieval.

    Pipeline:

        query text
            ↓
        embedding model
            ↓
        query embedding
            ↓
        pgvector similarity search
            ↓
        similarity threshold
            ↓
        relevant chunks
    """

    def __init__(
        self,
        connection: Connection,
        embedding_service: EmbeddingService | None = None,
        min_score: float | None = None,
    ) -> None:
        self.repository = DocumentRepository(connection)

        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else get_embedding_service()
        )

        settings = get_settings()

        self.min_score = (
            min_score
            if min_score is not None
            else settings.retrieval_min_score
        )

        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(
                "min_score must be between 0.0 and 1.0."
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_embedding = self.embedding_service.embed_text(
            query
        )

        results = self.repository.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
        )

        formatted_results: list[dict[str, Any]] = []

        for result in results:
            distance = float(result["distance"])
            similarity_score = 1.0 - distance

            if similarity_score < self.min_score:
                continue

            formatted_results.append(
                {
                    "chunk_id": result["id"],
                    "document_id": result["document_id"],
                    "page_number": result["page_number"],
                    "chunk_index": result["chunk_index"],
                    "text": result["text"],
                    "character_count": result["character_count"],
                    "score": similarity_score,
                    "distance": distance,
                }
            )

        return formatted_results

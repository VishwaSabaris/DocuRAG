import pytest
from uuid import uuid4

from app.db.database import pool
from app.services.retrieval.retrieval_service import (
    RetrievalService,
)


class FakeEmbeddingService:
    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        return [1.0] + [0.0] * 383


def test_retrieval_service() -> None:
    document_id = uuid4()

    chunk_ids = [
        uuid4(),
        uuid4(),
        uuid4(),
    ]

    with pool.connection() as connection:
        retrieval_service = RetrievalService(
            connection=connection,
            embedding_service=FakeEmbeddingService(),
            min_score=0.30,
        )

        repository = retrieval_service.repository

        repository.create_document(
            document_id=document_id,
            filename="retrieval-test.pdf",
            stored_filename=(
                f"{document_id}_retrieval-test.pdf"
            ),
        )

        chunks = [
            {
                "id": chunk_ids[0],
                "document_id": document_id,
                "page_number": 1,
                "chunk_index": 0,
                "text": "Python programming language.",
                "character_count": 29,
            },
            {
                "id": chunk_ids[1],
                "document_id": document_id,
                "page_number": 1,
                "chunk_index": 1,
                "text": "Cloud infrastructure and servers.",
                "character_count": 34,
            },
            {
                "id": chunk_ids[2],
                "document_id": document_id,
                "page_number": 2,
                "chunk_index": 2,
                "text": (
                    "Machine learning and "
                    "artificial intelligence."
                ),
                "character_count": 47,
            },
        ]

        repository.save_chunks(chunks)

        embeddings = [
            {
                "chunk_id": chunk_ids[0],
                "embedding": [1.0] + [0.0] * 383,
            },
            {
                "chunk_id": chunk_ids[1],
                "embedding": (
                    [0.0, 1.0] + [0.0] * 382
                ),
            },
            {
                "chunk_id": chunk_ids[2],
                "embedding": (
                    [0.0, 0.0, 1.0] + [0.0] * 381
                ),
            },
        ]

        repository.save_chunk_embeddings(
            embeddings
        )

        results = retrieval_service.retrieve(
            query="Python programming",
            top_k=2,
            document_id=document_id,
        )

        # Only the Python chunk should pass
        # the minimum similarity threshold.
        assert len(results) == 1

        assert results[0]["chunk_id"] == chunk_ids[0]
        assert results[0]["score"] >= 0.30
        assert results[0]["score"] == 1.0

        for result in results:
            assert result["score"] >= 0.30
            assert "text" in result
            assert "page_number" in result
            assert "distance" in result

        deleted = repository.delete_document(
            document_id
        )

        assert deleted is True

        assert repository.get_document(
            document_id
        ) is None


def test_retrieval_service_rejects_invalid_threshold() -> None:
    with pool.connection() as connection:
        with pytest.raises(
            ValueError,
            match="min_score must be between 0.0 and 1.0.",
        ):
            RetrievalService(
                connection=connection,
                embedding_service=FakeEmbeddingService(),
                min_score=1.1,
            )

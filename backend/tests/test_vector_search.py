from uuid import uuid4

from app.db.database import pool
from app.repositories.document_repository import DocumentRepository


def test_similarity_search() -> None:
    document_id = uuid4()

    chunk_ids = [
        uuid4(),
        uuid4(),
        uuid4(),
    ]

    with pool.connection() as connection:
        repository = DocumentRepository(connection)

        repository.create_document(
            document_id=document_id,
            filename="vector-test.pdf",
            stored_filename=(
                f"{document_id}_vector-test.pdf"
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

        query_embedding = [1.0] + [0.0] * 383

        results = repository.similarity_search(
            query_embedding=query_embedding,
            top_k=2,
            document_id=document_id,
        )

        assert len(results) == 2

        assert results[0]["id"] == chunk_ids[0]
        assert results[0]["distance"] < 0.01

        assert results[1]["id"] != chunk_ids[0]

        for result in results:
            assert "text" in result
            assert "page_number" in result
            assert "distance" in result

        # Clean up test data.
        # document_chunks are deleted automatically
        # through ON DELETE CASCADE.
        deleted = repository.delete_document(
            document_id
        )

        assert deleted is True

        assert repository.get_document(
            document_id
        ) is None

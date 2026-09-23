from uuid import uuid4

from app.db.database import pool
from app.repositories.document_repository import DocumentRepository


def test_document_repository() -> None:
    document_id = uuid4()
    chunk_ids = [uuid4(), uuid4()]

    with pool.connection() as connection:
        repository = DocumentRepository(connection)

        document = repository.create_document(
            document_id=document_id,
            filename="test.pdf",
            stored_filename=f"{document_id}_test.pdf",
        )

        assert document["id"] == document_id
        assert document["filename"] == "test.pdf"
        assert document["status"] == "uploaded"

        chunks = [
            {
                "id": chunk_ids[0],
                "document_id": document_id,
                "page_number": 1,
                "chunk_index": 0,
                "text": "This is a test document chunk.",
                "character_count": 31,
            },
            {
                "id": chunk_ids[1],
                "document_id": document_id,
                "page_number": 1,
                "chunk_index": 1,
                "text": "This is the second test chunk.",
                "character_count": 31,
            },
        ]

        saved_count = repository.save_chunks(chunks)

        assert saved_count == 2

        embeddings = [
            {
                "chunk_id": chunk_ids[0],
                "embedding": [0.1] * 384,
            },
            {
                "chunk_id": chunk_ids[1],
                "embedding": [0.2] * 384,
            },
        ]

        updated_count = repository.save_chunk_embeddings(
            embeddings
        )

        assert updated_count == 2

        stored_chunks = repository.get_document_chunks(
            document_id
        )

        assert len(stored_chunks) == 2

        assert stored_chunks[0]["chunk_index"] == 0
        assert stored_chunks[1]["chunk_index"] == 1

        assert stored_chunks[0]["embedding"] is not None
        assert stored_chunks[1]["embedding"] is not None

        repository.update_document_status(
            document_id=document_id,
            status="processed",
        )

        updated_document = repository.get_document(
            document_id
        )

        assert updated_document is not None
        assert updated_document["status"] == "processed"

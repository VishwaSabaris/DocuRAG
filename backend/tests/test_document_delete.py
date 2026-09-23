from uuid import uuid4

from app.db.database import pool
from app.repositories.document_repository import DocumentRepository


def test_delete_document() -> None:
    document_id = uuid4()

    with pool.connection() as connection:
        repository = DocumentRepository(connection)

        repository.create_document(
            document_id=document_id,
            filename="delete-test.pdf",
            stored_filename=f"{document_id}_delete-test.pdf",
        )

        document = repository.get_document(
            document_id
        )

        assert document is not None

        deleted = repository.delete_document(
            document_id
        )

        assert deleted is True

        document = repository.get_document(
            document_id
        )

        assert document is None

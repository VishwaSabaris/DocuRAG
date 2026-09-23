from uuid import UUID

import pytest

from app.db.database import pool
from app.repositories.document_repository import DocumentRepository


DOCUMENT_ID = UUID(
    "8aedc5d9-6e1b-4d61-8c17-33553e0896f3"
)


def test_keyword_search_finds_exact_name() -> None:
    with pool.connection() as connection:
        repository = DocumentRepository(connection)

        results = repository.keyword_search(
            query_text="Vishwa Sabaris V",
            top_k=5,
            document_id=DOCUMENT_ID,
        )

    assert results

    assert any(
        "Vishwa Sabaris V" in result["text"]
        for result in results
    )


def test_keyword_search_finds_exact_id() -> None:
    with pool.connection() as connection:
        repository = DocumentRepository(connection)

        results = repository.keyword_search(
            query_text="NOC26CS117S353802605",
            top_k=5,
            document_id=DOCUMENT_ID,
        )

    assert results

    assert any(
        "NOC26CS117S353802605" in result["text"]
        for result in results
    )


def test_keyword_search_is_case_insensitive() -> None:
    with pool.connection() as connection:
        repository = DocumentRepository(connection)

        results = repository.keyword_search(
            query_text="vishwa sabaris v",
            top_k=5,
            document_id=DOCUMENT_ID,
        )

    assert results

    assert any(
        "Vishwa Sabaris V" in result["text"]
        for result in results
    )


def test_keyword_search_returns_empty_for_missing_text() -> None:
    with pool.connection() as connection:
        repository = DocumentRepository(connection)

        results = repository.keyword_search(
            query_text="This Text Does Not Exist In The Document",
            top_k=5,
            document_id=DOCUMENT_ID,
        )

    assert results == []


def test_keyword_search_rejects_empty_query() -> None:
    with pool.connection() as connection:
        repository = DocumentRepository(connection)

        with pytest.raises(
            ValueError,
            match="Query text cannot be empty.",
        ):
            repository.keyword_search(
                query_text="",
                top_k=5,
                document_id=DOCUMENT_ID,
            )


def test_keyword_search_rejects_invalid_top_k() -> None:
    with pool.connection() as connection:
        repository = DocumentRepository(connection)

        with pytest.raises(
            ValueError,
            match="top_k must be greater than 0.",
        ):
            repository.keyword_search(
                query_text="Vishwa Sabaris V",
                top_k=0,
                document_id=DOCUMENT_ID,
            )

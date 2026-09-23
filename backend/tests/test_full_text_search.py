from uuid import uuid4

import pytest

from app.repositories.document_repository import DocumentRepository


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed_query = None
        self.executed_parameters = None

    def execute(self, query, parameters):
        self.executed_query = query
        self.executed_parameters = parameters

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeConnection:
    def __init__(self, rows=None):
        self.cursor_instance = FakeCursor(rows)

    def cursor(self):
        return self.cursor_instance


def test_full_text_search_returns_results() -> None:
    rows = [
        {
            "id": uuid4(),
            "document_id": uuid4(),
            "page_number": 1,
            "chunk_index": 0,
            "text": "Vishwa Sabaris V",
            "character_count": 16,
            "lexical_score": 0.1,
        }
    ]

    connection = FakeConnection(rows)
    repository = DocumentRepository(connection)

    result = repository.full_text_search(
        query_text="Vishwa Sabaris V",
        top_k=5,
    )

    assert result == rows
    assert "search_vector" in connection.cursor_instance.executed_query
    assert "ts_rank_cd" in connection.cursor_instance.executed_query
    assert "LIMIT %s" in connection.cursor_instance.executed_query


def test_full_text_search_filters_by_document() -> None:
    document_id = uuid4()

    connection = FakeConnection([])
    repository = DocumentRepository(connection)

    repository.full_text_search(
        query_text="examination date",
        top_k=5,
        document_id=document_id,
    )

    query = connection.cursor_instance.executed_query
    parameters = connection.cursor_instance.executed_parameters

    assert "document_id = %s" in query
    assert parameters[-2] == document_id
    assert parameters[-1] == 5


def test_full_text_search_rejects_empty_query() -> None:
    connection = FakeConnection()
    repository = DocumentRepository(connection)

    with pytest.raises(ValueError, match="Query text cannot be empty"):
        repository.full_text_search("   ")


def test_full_text_search_rejects_invalid_top_k() -> None:
    connection = FakeConnection()
    repository = DocumentRepository(connection)

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        repository.full_text_search(
            query_text="test",
            top_k=0,
        )

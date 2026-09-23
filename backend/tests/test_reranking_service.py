from app.services.reranking.reranking_service import (
    RerankingService,
)


class FakeCrossEncoder:
    def predict(
        self,
        pairs,
        show_progress_bar=False,
    ):
        scores = []

        for _, document in pairs:
            if "examination date" in document.lower():
                scores.append(9.0)
            elif "examination center" in document.lower():
                scores.append(5.0)
            else:
                scores.append(1.0)

        return scores


def test_reranking_orders_documents_by_score() -> None:
    service = object.__new__(
        RerankingService
    )

    service.model = FakeCrossEncoder()

    documents = [
        {
            "chunk_id": "chunk-1",
            "text": "This document describes the examination center.",
        },
        {
            "chunk_id": "chunk-2",
            "text": "The examination date is 20 September, 2026.",
        },
        {
            "chunk_id": "chunk-3",
            "text": "This document contains unrelated information.",
        },
    ]

    results = service.rerank(
        query="What is the examination date?",
        documents=documents,
        top_k=3,
    )

    assert len(results) == 3

    assert (
        results[0]["chunk_id"]
        == "chunk-2"
    )

    assert (
        results[1]["chunk_id"]
        == "chunk-1"
    )

    assert (
        results[2]["chunk_id"]
        == "chunk-3"
    )

    assert (
        results[0]["rerank_score"]
        > results[1]["rerank_score"]
    )


def test_reranking_limits_top_k() -> None:
    service = object.__new__(
        RerankingService
    )

    service.model = FakeCrossEncoder()

    documents = [
        {
            "chunk_id": f"chunk-{index}",
            "text": f"Document {index}",
        }
        for index in range(5)
    ]

    results = service.rerank(
        query="document",
        documents=documents,
        top_k=2,
    )

    assert len(results) == 2


def test_reranking_empty_documents() -> None:
    service = object.__new__(
        RerankingService
    )

    service.model = FakeCrossEncoder()

    results = service.rerank(
        query="What is the examination date?",
        documents=[],
        top_k=5,
    )

    assert results == []


def test_reranking_rejects_empty_query() -> None:
    service = object.__new__(
        RerankingService
    )

    service.model = FakeCrossEncoder()

    try:
        service.rerank(
            query="",
            documents=[],
            top_k=5,
        )
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty."
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_reranking_rejects_invalid_top_k() -> None:
    service = object.__new__(
        RerankingService
    )

    service.model = FakeCrossEncoder()

    try:
        service.rerank(
            query="test",
            documents=[],
            top_k=0,
        )
    except ValueError as exc:
        assert (
            str(exc)
            == "top_k must be greater than 0."
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )

from uuid import UUID

import pytest

from app.db.database import pool
from app.services.retrieval.hybrid_retrieval_service import (
    HybridRetrievalService,
)


DOCUMENT_ID = UUID(
    "8aedc5d9-6e1b-4d61-8c17-33553e0896f3"
)


def test_hybrid_retrieval_finds_exact_name() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        results = service.retrieve(
            query="Vishwa Sabaris V",
            vector_top_k=10,
            keyword_top_k=10,
            full_text_top_k=10,
            document_id=DOCUMENT_ID,
        )

    assert results

    matching_results = [
        result
        for result in results
        if "Vishwa Sabaris V" in result["text"]
    ]

    assert matching_results

    result = matching_results[0]

    assert "keyword" in result["retrieval_methods"]
    assert "full_text" in result["retrieval_methods"]
    assert result["keyword_score"] == 1.0


def test_hybrid_retrieval_finds_exact_id() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        results = service.retrieve(
            query="NOC26CS117S353802605",
            vector_top_k=10,
            keyword_top_k=10,
            full_text_top_k=10,
            document_id=DOCUMENT_ID,
        )

    assert results

    matching_results = [
        result
        for result in results
        if "NOC26CS117S353802605" in result["text"]
    ]

    assert matching_results

    result = matching_results[0]

    assert "keyword" in result["retrieval_methods"]
    assert "full_text" in result["retrieval_methods"]
    assert result["keyword_score"] == 1.0


def test_hybrid_retrieval_supports_semantic_questions() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        results = service.retrieve(
            query="What is the examination date?",
            vector_top_k=10,
            keyword_top_k=10,
            full_text_top_k=10,
            document_id=DOCUMENT_ID,
        )

    assert results

    assert any(
        "20 September, 2026" in result["text"]
        for result in results
    )


def test_hybrid_retrieval_merges_duplicate_candidates() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        results = service.retrieve(
            query="Vishwa Sabaris V",
            vector_top_k=10,
            keyword_top_k=10,
            full_text_top_k=10,
            document_id=DOCUMENT_ID,
        )

    chunk_ids = [
        result["chunk_id"]
        for result in results
    ]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_hybrid_retrieval_tracks_retrieval_methods() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        results = service.retrieve(
            query="Vishwa Sabaris V",
            vector_top_k=10,
            keyword_top_k=10,
            full_text_top_k=10,
            document_id=DOCUMENT_ID,
        )

    assert results

    result = next(
        result
        for result in results
        if "Vishwa Sabaris V" in result["text"]
    )

    assert isinstance(
        result["retrieval_methods"],
        list,
    )

    assert "keyword" in result["retrieval_methods"]
    assert "full_text" in result["retrieval_methods"]


def test_hybrid_retrieval_tracks_rrf_score() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        results = service.retrieve(
            query="Vishwa Sabaris V",
            vector_top_k=10,
            keyword_top_k=10,
            full_text_top_k=10,
            document_id=DOCUMENT_ID,
        )

    assert results

    result = next(
        result
        for result in results
        if "Vishwa Sabaris V" in result["text"]
    )

    assert result["rrf_score"] > 0.0
    assert isinstance(
        result["retrieval_ranks"],
        dict,
    )


def test_hybrid_retrieval_tracks_multiple_retrieval_ranks() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        results = service.retrieve(
            query="Vishwa Sabaris V",
            vector_top_k=10,
            keyword_top_k=10,
            full_text_top_k=10,
            document_id=DOCUMENT_ID,
        )

    result = next(
        result
        for result in results
        if "Vishwa Sabaris V" in result["text"]
    )

    assert "keyword" in result["retrieval_ranks"]
    assert "full_text" in result["retrieval_ranks"]


def test_hybrid_retrieval_does_not_claim_keyword_match_for_missing_information() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        results = service.retrieve(
            query="This information definitely does not exist",
            vector_top_k=10,
            keyword_top_k=10,
            full_text_top_k=10,
            document_id=DOCUMENT_ID,
        )

    for result in results:
        assert "keyword" not in result["retrieval_methods"]
        assert result["keyword_score"] == 0.0


def test_hybrid_retrieval_rejects_empty_query() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        with pytest.raises(
            ValueError,
            match="Query cannot be empty.",
        ):
            service.retrieve(
                query="",
                document_id=DOCUMENT_ID,
            )


def test_hybrid_retrieval_rejects_invalid_vector_top_k() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        with pytest.raises(
            ValueError,
            match="vector_top_k must be greater than 0.",
        ):
            service.retrieve(
                query="Vishwa Sabaris V",
                vector_top_k=0,
                document_id=DOCUMENT_ID,
            )


def test_hybrid_retrieval_rejects_invalid_keyword_top_k() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        with pytest.raises(
            ValueError,
            match="keyword_top_k must be greater than 0.",
        ):
            service.retrieve(
                query="Vishwa Sabaris V",
                keyword_top_k=0,
                document_id=DOCUMENT_ID,
            )


def test_hybrid_retrieval_rejects_invalid_full_text_top_k() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        with pytest.raises(
            ValueError,
            match="full_text_top_k must be greater than 0.",
        ):
            service.retrieve(
                query="Vishwa Sabaris V",
                full_text_top_k=0,
                document_id=DOCUMENT_ID,
            )


def test_rrf_score_uses_one_based_rank() -> None:
    rank_one = HybridRetrievalService._rrf_score(1)
    rank_two = HybridRetrievalService._rrf_score(2)

    assert rank_one == pytest.approx(
        1.0 / 61.0
    )

    assert rank_two == pytest.approx(
        1.0 / 62.0
    )

    assert rank_one > rank_two


def test_rrf_score_rejects_invalid_rank() -> None:
    with pytest.raises(
        ValueError,
        match="RRF rank must be greater than 0.",
    ):
        HybridRetrievalService._rrf_score(0)


def test_hybrid_retrieval_applies_exact_match_boost() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        results = service.retrieve(
            query="NOC26CS117S353802605",
            vector_top_k=20,
            keyword_top_k=20,
            full_text_top_k=20,
            document_id=DOCUMENT_ID,
        )

    assert results

    result = next(
        result
        for result in results
        if "NOC26CS117S353802605" in result["text"]
    )

    assert result["exact_match_score"] == 1.0
    assert (
        result["exact_match_boost"]
        == HybridRetrievalService.EXACT_MATCH_BOOST
    )
    assert result["hybrid_score"] == pytest.approx(
        result["rrf_score"]
        + result["exact_match_boost"]
    )


def test_hybrid_retrieval_does_not_apply_exact_match_boost_to_unmatched_query() -> None:
    with pool.connection() as connection:
        service = HybridRetrievalService(
            connection=connection
        )

        results = service.retrieve(
            query="This information definitely does not exist",
            vector_top_k=20,
            keyword_top_k=20,
            full_text_top_k=20,
            document_id=DOCUMENT_ID,
        )

    for result in results:
        assert result["exact_match_score"] == 0.0
        assert result["exact_match_boost"] == 0.0
        assert result["hybrid_score"] == pytest.approx(
            result["rrf_score"]
        )


def test_exact_match_detection_is_case_insensitive() -> None:
    assert HybridRetrievalService._has_exact_match(
        query="Vishwa Sabaris V",
        text="Candidate Name: vishwa sabaris v",
    )


def test_exact_match_detection_normalizes_whitespace() -> None:
    assert HybridRetrievalService._has_exact_match(
        query="Vishwa   Sabaris V",
        text="Candidate Name: Vishwa Sabaris V",
    )


def test_exact_match_detection_rejects_missing_text() -> None:
    assert not HybridRetrievalService._has_exact_match(
        query="Vishwa Sabaris V",
        text="Candidate Name: Another Person",
    )

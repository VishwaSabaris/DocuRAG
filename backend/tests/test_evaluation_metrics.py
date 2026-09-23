from evaluation.metrics import (
    calculate_average_score,
    calculate_case_metrics,
    calculate_content_match,
    calculate_dataset_metrics,
    calculate_page_accuracy,
    calculate_refusal_accuracy,
)


def test_content_match() -> None:
    assert calculate_content_match(
        expected_answer=(
            "Sunday, 20 September, 2026"
        ),
        actual_answer=(
            "The examination date is "
            "Sunday, 20 September, 2026."
        ),
        expected_behavior="answer",
    )


def test_content_match_fails_for_wrong_answer() -> None:
    assert not calculate_content_match(
        expected_answer=(
            "Sunday, 20 September, 2026"
        ),
        actual_answer="The exam is in October.",
        expected_behavior="answer",
    )


def test_refusal_content_match() -> None:
    assert calculate_content_match(
        expected_answer=(
            "The information is not available "
            "in the provided document."
        ),
        actual_answer=(
            "The information is not available "
            "in the provided document."
        ),
        expected_behavior="refuse",
    )


def test_page_accuracy() -> None:
    assert calculate_page_accuracy(
        expected_pages=[1],
        source_pages=[1],
    )


def test_page_accuracy_fails() -> None:
    assert not calculate_page_accuracy(
        expected_pages=[1],
        source_pages=[2],
    )


def test_empty_page_accuracy() -> None:
    assert calculate_page_accuracy(
        expected_pages=[],
        source_pages=[],
    )


def test_refusal_accuracy() -> None:
    assert calculate_refusal_accuracy(
        expected_behavior="refuse",
        actual_behavior="refuse",
    )

    assert not calculate_refusal_accuracy(
        expected_behavior="refuse",
        actual_behavior="answer",
    )


def test_average_score() -> None:
    score = calculate_average_score(
        [0.4, 0.6, 0.8]
    )

    assert score == 0.6


def test_case_metrics() -> None:
    result = calculate_case_metrics(
        {
            "expected_behavior": "answer",
            "actual_behavior": "answer",
            "expected_answer": (
                "Sunday, 20 September, 2026"
            ),
            "answer": (
                "Sunday, 20 September, 2026"
            ),
            "expected_pages": [1],
            "source_pages": [1],
            "retrieval_scores": [0.44],
        }
    )

    assert result["content_match"] is True
    assert result["page_accuracy"] is True
    assert result["refusal_accuracy"] is True
    assert result["average_retrieval_score"] == 0.44


def test_dataset_metrics() -> None:
    results = [
        {
            "expected_behavior": "answer",
            "actual_behavior": "answer",
            "expected_answer": (
                "Sunday, 20 September, 2026"
            ),
            "answer": (
                "Sunday, 20 September, 2026"
            ),
            "expected_pages": [1],
            "source_pages": [1],
            "retrieval_scores": [0.44],
            "response_time_seconds": 2.0,
        },
        {
            "expected_behavior": "refuse",
            "actual_behavior": "refuse",
            "expected_answer": (
                "The information is not available "
                "in the provided document."
            ),
            "answer": (
                "The information is not available "
                "in the provided document."
            ),
            "expected_pages": [],
            "source_pages": [],
            "retrieval_scores": [],
            "response_time_seconds": 1.0,
        },
    ]

    metrics = calculate_dataset_metrics(
        results
    )

    assert metrics["total_cases"] == 2
    assert metrics["behavior_accuracy"] == 1.0
    assert metrics["content_match_accuracy"] == 1.0
    assert metrics["page_accuracy"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0
    assert metrics["average_retrieval_score"] == 0.44
    assert (
        metrics["average_response_time_seconds"]
        == 1.5
    )

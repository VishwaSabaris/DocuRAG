from typing import Any


REFUSAL_TEXT = (
    "The information is not available "
    "in the provided document."
)


def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic comparison.
    """
    return " ".join(
        text.lower().strip().split()
    )


def calculate_content_match(
    expected_answer: str,
    actual_answer: str,
    expected_behavior: str,
) -> bool:
    """
    Check whether the expected answer content
    is present in the generated answer.

    For refusal cases, verify the expected refusal
    message.
    """

    expected = normalize_text(
        expected_answer
    )

    actual = normalize_text(
        actual_answer
    )

    if expected_behavior == "refuse":
        return (
            expected in actual
            or normalize_text(REFUSAL_TEXT) in actual
        )

    return expected in actual


def calculate_page_accuracy(
    expected_pages: list[int],
    source_pages: list[int],
) -> bool:
    """
    Verify that all expected source pages are present
    in the retrieved source pages.
    """

    if not expected_pages:
        return not source_pages

    return all(
        page in source_pages
        for page in expected_pages
    )


def calculate_refusal_accuracy(
    expected_behavior: str,
    actual_behavior: str,
) -> bool:
    """
    Check whether the system correctly answered
    or refused.
    """

    return (
        expected_behavior
        == actual_behavior
    )


def calculate_average_score(
    retrieval_scores: list[float],
) -> float | None:
    """
    Calculate the average similarity score
    for retrieved chunks.
    """

    if not retrieval_scores:
        return None

    return sum(retrieval_scores) / len(
        retrieval_scores
    )


def calculate_case_metrics(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate deterministic evaluation metrics
    for a single evaluation case.
    """

    expected_behavior = result[
        "expected_behavior"
    ]

    actual_behavior = result[
        "actual_behavior"
    ]

    expected_answer = result.get(
        "expected_answer",
        "",
    )

    actual_answer = result.get(
        "answer",
        "",
    )

    expected_pages = result.get(
        "expected_pages",
        [],
    )

    source_pages = result.get(
        "source_pages",
        [],
    )

    retrieval_scores = result.get(
        "retrieval_scores",
        [],
    )

    content_match = calculate_content_match(
        expected_answer=expected_answer,
        actual_answer=actual_answer,
        expected_behavior=expected_behavior,
    )

    page_accuracy = calculate_page_accuracy(
        expected_pages=expected_pages,
        source_pages=source_pages,
    )

    refusal_accuracy = calculate_refusal_accuracy(
        expected_behavior=expected_behavior,
        actual_behavior=actual_behavior,
    )

    average_score = calculate_average_score(
        retrieval_scores
    )

    return {
        "content_match": content_match,
        "page_accuracy": page_accuracy,
        "refusal_accuracy": refusal_accuracy,
        "average_retrieval_score": average_score,
    }


def calculate_dataset_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate aggregate metrics for the complete
    evaluation dataset.
    """

    if not results:
        return {
            "total_cases": 0,
            "behavior_accuracy": 0.0,
            "content_match_accuracy": 0.0,
            "page_accuracy": 0.0,
            "refusal_accuracy": 0.0,
            "average_retrieval_score": None,
            "average_response_time_seconds": 0.0,
        }

    case_metrics = [
        calculate_case_metrics(result)
        for result in results
    ]

    total_cases = len(results)

    behavior_accuracy = sum(
        result["refusal_accuracy"]
        for result in case_metrics
    ) / total_cases

    content_match_accuracy = sum(
        result["content_match"]
        for result in case_metrics
    ) / total_cases

    page_accuracy = sum(
        result["page_accuracy"]
        for result in case_metrics
    ) / total_cases

    refusal_accuracy = sum(
        result["refusal_accuracy"]
        for result in case_metrics
    ) / total_cases

    scores = [
        result["average_retrieval_score"]
        for result in case_metrics
        if result["average_retrieval_score"]
        is not None
    ]

    average_retrieval_score = (
        sum(scores) / len(scores)
        if scores
        else None
    )

    response_times = [
        float(
            result.get(
                "response_time_seconds",
                0.0,
            )
        )
        for result in results
    ]

    average_response_time = (
        sum(response_times)
        / len(response_times)
    )

    return {
        "total_cases": total_cases,
        "behavior_accuracy": behavior_accuracy,
        "content_match_accuracy": (
            content_match_accuracy
        ),
        "page_accuracy": page_accuracy,
        "refusal_accuracy": refusal_accuracy,
        "average_retrieval_score": (
            average_retrieval_score
        ),
        "average_response_time_seconds": (
            average_response_time
        ),
    }

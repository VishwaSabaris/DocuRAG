import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from evaluation.metrics import calculate_dataset_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset.json"
)

BASE_URL = "http://127.0.0.1:8000"

REQUEST_TIMEOUT = 120.0


def load_dataset() -> dict[str, Any]:
    """
    Load the evaluation dataset from disk.
    """

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def classify_behavior(
    answer: str,
    sources: list[dict[str, Any]],
) -> str:
    """
    Determine whether the system answered or refused.

    The application returns an empty source list when
    retrieval finds no relevant document context.
    """

    if not sources:
        return "refuse"

    return "answer"


def evaluate_case(
    client: httpx.Client,
    document_id: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute one evaluation case against the running API.
    """

    question = case["question"]

    endpoint = (
        f"{BASE_URL}/documents/"
        f"{document_id}/ask"
    )

    start_time = time.perf_counter()

    try:
        response = client.post(
            endpoint,
            json={
                "query": question,
                "top_k": 3,
            },
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

    except httpx.HTTPError as exc:
        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "case_id": case["id"],
            "question": question,
            "expected_behavior": case[
                "expected_behavior"
            ],
            "actual_behavior": "error",
            "expected_answer": case.get(
                "expected_answer",
                "",
            ),
            "answer": "",
            "expected_pages": case.get(
                "expected_pages",
                [],
            ),
            "source_pages": [],
            "retrieval_scores": [],
            "rerank_scores": [],
            "response_time_seconds": elapsed,
            "error": str(exc),
        }

    if response.status_code != 200:
        return {
            "case_id": case["id"],
            "question": question,
            "expected_behavior": case[
                "expected_behavior"
            ],
            "actual_behavior": "error",
            "expected_answer": case.get(
                "expected_answer",
                "",
            ),
            "answer": "",
            "expected_pages": case.get(
                "expected_pages",
                [],
            ),
            "source_pages": [],
            "retrieval_scores": [],
            "rerank_scores": [],
            "response_time_seconds": elapsed,
            "error": (
                f"HTTP {response.status_code}: "
                f"{response.text}"
            ),
        }

    data = response.json()

    answer = data.get(
        "answer",
        "",
    )

    sources = data.get(
        "sources",
        [],
    )

    actual_behavior = classify_behavior(
        answer=answer,
        sources=sources,
    )

    source_pages = [
        int(source["page_number"])
        for source in sources
        if "page_number" in source
    ]

    retrieval_scores = [
        float(source["score"])
        for source in sources
        if "score" in source
    ]

    rerank_scores = [
        float(source["rerank_score"])
        for source in sources
        if source.get("rerank_score") is not None
    ]

    return {
        "case_id": case["id"],
        "question": question,
        "expected_behavior": case[
            "expected_behavior"
        ],
        "actual_behavior": actual_behavior,
        "expected_answer": case.get(
            "expected_answer",
            "",
        ),
        "answer": answer,
        "expected_pages": case.get(
            "expected_pages",
            [],
        ),
        "source_pages": source_pages,
        "retrieval_scores": retrieval_scores,
        "rerank_scores": rerank_scores,
        "response_time_seconds": elapsed,
    }


def print_case_result(
    result: dict[str, Any],
) -> None:
    """
    Print detailed evaluation results for one case.
    """

    from evaluation.metrics import calculate_case_metrics

    metrics = calculate_case_metrics(result)

    expected_behavior = result[
        "expected_behavior"
    ]

    actual_behavior = result[
        "actual_behavior"
    ]

    if result.get("error"):
        print(
            f"[ERROR] {result['case_id']}: "
            f"{result['question']}"
        )
        print(
            f"      {result['error']}"
        )
        print()
        return

    overall_pass = (
        metrics["content_match"]
        and metrics["page_accuracy"]
        and metrics["refusal_accuracy"]
    )

    print(
        f"[{'PASS' if overall_pass else 'FAIL'}] "
        f"{result['case_id']}: "
        f"{result['question']}"
    )

    print(
        f"      Expected behavior: "
        f"{expected_behavior}"
    )

    print(
        f"      Actual behavior:   "
        f"{actual_behavior}"
    )

    print(
        f"      Content match:     "
        f"{'PASS' if metrics['content_match'] else 'FAIL'}"
    )

    print(
        f"      Page accuracy:     "
        f"{'PASS' if metrics['page_accuracy'] else 'FAIL'}"
    )

    print(
        f"      Time:               "
        f"{result['response_time_seconds']:.3f}s"
    )

    retrieval_scores = result.get(
        "retrieval_scores",
        [],
    )

    if retrieval_scores:
        print(
            f"      Vector scores:       "
            f"{[
                round(score, 4)
                for score in retrieval_scores
            ]}"
        )

    rerank_scores = result.get(
        "rerank_scores",
        [],
    )

    if rerank_scores:
        print(
            f"      Rerank scores:       "
            f"{[
                round(score, 4)
                for score in rerank_scores
            ]}"
        )

    print(
        f"      Sources:             "
        f"{len(result.get('source_pages', []))}"
    )

    print(
        f"      Answer:             "
        f"{result.get('answer', '')}"
    )

    print()


def print_summary(
    metrics: dict[str, Any],
) -> None:
    """
    Print aggregate evaluation metrics.
    """

    print("=" * 70)
    print("Evaluation Summary")
    print("=" * 70)

    print(
        f"Total Cases:              "
        f"{metrics['total_cases']}"
    )

    print(
        f"Behavior Accuracy:        "
        f"{metrics['behavior_accuracy']:.2%}"
    )

    print(
        f"Content Match Accuracy:   "
        f"{metrics['content_match_accuracy']:.2%}"
    )

    print(
        f"Page Accuracy:             "
        f"{metrics['page_accuracy']:.2%}"
    )

    print(
        f"Refusal Accuracy:          "
        f"{metrics['refusal_accuracy']:.2%}"
    )

    average_score = metrics[
        "average_retrieval_score"
    ]

    if average_score is None:
        print(
            "Average Retrieval Score:  N/A"
        )
    else:
        print(
            f"Average Retrieval Score:  "
            f"{average_score:.4f}"
        )

    print(
        f"Average Response Time:     "
        f"{metrics['average_response_time_seconds']:.3f}s"
    )

    print("=" * 70)


def main() -> int:
    """
    Run the complete DocuRAG evaluation.
    """

    dataset = load_dataset()

    document_id = dataset[
        "document_id"
    ]

    cases = dataset[
        "cases"
    ]

    print("=" * 70)
    print("DocuRAG Evaluation")
    print("=" * 70)
    print()

    results: list[dict[str, Any]] = []

    with httpx.Client(
        timeout=REQUEST_TIMEOUT
    ) as client:

        for case in cases:
            result = evaluate_case(
                client=client,
                document_id=document_id,
                case=case,
            )

            results.append(result)

            print_case_result(result)

    metrics = calculate_dataset_metrics(
        results
    )

    print_summary(metrics)

    behavior_accuracy = metrics[
        "behavior_accuracy"
    ]

    content_accuracy = metrics[
        "content_match_accuracy"
    ]

    page_accuracy = metrics[
        "page_accuracy"
    ]

    all_metrics_pass = (
        behavior_accuracy == 1.0
        and content_accuracy == 1.0
        and page_accuracy == 1.0
    )

    print()

    if all_metrics_pass:
        print(
            "Evaluation Result: PASS"
        )
        return 0

    print(
        "Evaluation Result: FAIL"
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())

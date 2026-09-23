import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import httpx


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_PATH = (
    BASE_DIR
    / "evaluation"
    / "datasets"
    / "nptel_exam.json"
)

DEFAULT_OUTPUT_PATH = (
    BASE_DIR
    / "evaluation"
    / "latest_results.json"
)

DEFAULT_DOCUMENT_ID = (
    "8aedc5d9-6e1b-4d61-8c17-33553e0896f3"
)

BASE_API_URL = "http://127.0.0.1:8000"

REFUSAL_MESSAGE = (
    "The information is not available in the provided document."
)


def normalize(text: str) -> str:
    return " ".join(
        str(text).strip().lower().split()
    )


def contains_expected_answer(
    answer: str,
    expected_answer: str | None,
) -> bool:
    if expected_answer is None:
        return False

    return normalize(expected_answer) in normalize(answer)


def contains_all_expected_fragments(
    answer: str,
    fragments: list[str],
) -> bool:
    normalized_answer = normalize(answer)

    return all(
        normalize(fragment) in normalized_answer
        for fragment in fragments
    )


def is_refusal(answer: str) -> bool:
    normalized = normalize(answer)

    return normalize(
        REFUSAL_MESSAGE
    ) in normalized


def extract_source_document_ids(
    sources: list,
) -> set[str]:
    document_ids: set[str] = set()

    for source in sources:
        if not isinstance(source, dict):
            continue

        for key in (
            "document_id",
            "documentId",
            "document",
        ):
            value = source.get(key)

            if value:
                document_ids.add(
                    str(value)
                )

    return document_ids


def check_retrieval(
    case: dict,
    sources: list,
) -> tuple[bool | None, str]:
    """
    Returns:

    True  -> retrieval expectation satisfied
    False -> retrieval expectation failed
    None  -> retrieval correctness not applicable
    """

    expected_retrieval = case.get(
        "expected_retrieval"
    )

    if expected_retrieval is None:
        expected_retrieval = (
            not case.get(
                "expected_refusal",
                False,
            )
        )

    if not expected_retrieval:
        return None, "not_applicable"

    if not sources:
        return False, "no_sources"

    expected_document_id = case.get(
        "expected_document_id"
    )

    if expected_document_id:
        source_document_ids = (
            extract_source_document_ids(
                sources
            )
        )

        if source_document_ids:
            if expected_document_id in source_document_ids:
                return True, "document_match"

            return False, "wrong_document"

    return True, "sources_present"


def evaluate_answer(
    case: dict,
    answer: str,
) -> tuple[bool, str]:
    """
    Evaluate the answer according to an explicit
    expected behavior.

    Supported behaviors:

    - answer_contains
    - answer_contains_all
    - refusal
    - answer_not_refusal
    """

    expected_behavior = case.get(
        "expected_behavior"
    )

    if expected_behavior is None:
        if case.get(
            "expected_refusal",
            False,
        ):
            expected_behavior = "refusal"

        elif case.get(
            "expected_answer"
        ) is not None:
            expected_behavior = (
                "answer_contains"
            )

        else:
            return (
                False,
                "missing_expected_behavior",
            )

    if expected_behavior == "refusal":
        if is_refusal(answer):
            return True, "refusal_match"

        return False, "expected_refusal"

    if expected_behavior == "answer_contains":
        expected_answer = case.get(
            "expected_answer"
        )

        if expected_answer is None:
            return (
                False,
                "missing_expected_answer",
            )

        if contains_expected_answer(
            answer,
            expected_answer,
        ):
            return True, "answer_match"

        return False, "answer_mismatch"

    if expected_behavior == "answer_contains_all":
        fragments = case.get(
            "expected_answer_fragments",
            [],
        )

        if not fragments:
            return (
                False,
                "missing_expected_fragments",
            )

        if contains_all_expected_fragments(
            answer,
            fragments,
        ):
            return True, "all_fragments_match"

        return False, "fragment_mismatch"

    if expected_behavior == "answer_not_refusal":
        if not is_refusal(answer):
            return True, "non_refusal_answer"

        return False, "unexpected_refusal"

    return (
        False,
        f"unknown_behavior:{expected_behavior}",
    )


def evaluate_case(
    client: httpx.Client,
    case: dict,
) -> dict:
    question = case["question"]

    document_id = case.get(
        "document_id",
        DEFAULT_DOCUMENT_ID,
    )

    api_url = (
        f"{BASE_API_URL}"
        f"/documents/{document_id}/ask"
    )

    started = time.perf_counter()

    try:
        response = client.post(
            api_url,
            json={
                "query": question,
                "top_k": case.get(
                    "top_k",
                    5,
                ),
            },
        )

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000

        response.raise_for_status()

        payload = response.json()

        answer = str(
            payload.get(
                "answer",
                "",
            )
        )

        sources = payload.get(
            "sources",
            [],
        )

        answer_correct, answer_reason = (
            evaluate_answer(
                case,
                answer,
            )
        )

        retrieval_correct, retrieval_reason = (
            check_retrieval(
                case,
                sources,
            )
        )

        passed = answer_correct

        if retrieval_correct is False:
            passed = False

        return {
            "id": case["id"],
            "type": case["type"],
            "question": question,
            "document_id": document_id,
            "expected_behavior": case.get(
                "expected_behavior"
            ),
            "expected_answer": case.get(
                "expected_answer"
            ),
            "expected_answer_fragments": case.get(
                "expected_answer_fragments"
            ),
            "answer": answer,
            "answer_correct": answer_correct,
            "answer_reason": answer_reason,
            "retrieval_correct": retrieval_correct,
            "retrieval_reason": retrieval_reason,
            "source_count": len(sources),
            "latency_ms": round(
                elapsed_ms,
                2,
            ),
            "passed": passed,
            "error": None,
        }

    except Exception as exc:
        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000

        return {
            "id": case["id"],
            "type": case["type"],
            "question": question,
            "document_id": document_id,
            "expected_behavior": case.get(
                "expected_behavior"
            ),
            "expected_answer": case.get(
                "expected_answer"
            ),
            "expected_answer_fragments": case.get(
                "expected_answer_fragments"
            ),
            "answer": "",
            "answer_correct": False,
            "answer_reason": "request_error",
            "retrieval_correct": False,
            "retrieval_reason": "request_error",
            "source_count": 0,
            "latency_ms": round(
                elapsed_ms,
                2,
            ),
            "passed": False,
            "error": str(exc),
        }


def print_result(result: dict) -> None:
    status = (
        "PASS"
        if result["passed"]
        else "FAIL"
    )

    print(
        f"[{status:4}] "
        f"{result['id']:20} "
        f"{result['latency_ms']:8.2f} ms"
    )

    if not result["passed"]:
        print(
            f"       Question: "
            f"{result['question']}"
        )

        print(
            f"       Answer:   "
            f"{result['answer']}"
        )

        if result.get("expected_answer"):
            print(
                f"       Expected: "
                f"{result['expected_answer']}"
            )

        if result.get(
            "expected_answer_fragments"
        ):
            print(
                f"       Fragments: "
                f"{result['expected_answer_fragments']}"
            )

        print(
            f"       Answer check: "
            f"{result['answer_reason']}"
        )

        print(
            f"       Retrieval check: "
            f"{result['retrieval_reason']}"
        )

        if result["error"]:
            print(
                f"       Error: "
                f"{result['error']}"
            )


def calculate_category_metrics(
    results: list[dict],
) -> dict:
    categories: dict[str, list[dict]] = (
        defaultdict(list)
    )

    for result in results:
        categories[
            result["type"]
        ].append(result)

    metrics = {}

    for category, category_results in (
        categories.items()
    ):
        total = len(category_results)

        passed = sum(
            1
            for result in category_results
            if result["passed"]
        )

        answer_correct = sum(
            1
            for result in category_results
            if result["answer_correct"]
        )

        retrieval_applicable = [
            result
            for result in category_results
            if result["retrieval_correct"]
            is not None
        ]

        retrieval_correct = sum(
            1
            for result in retrieval_applicable
            if result["retrieval_correct"]
        )

        latencies = [
            result["latency_ms"]
            for result in category_results
            if result["error"] is None
        ]

        metrics[category] = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "accuracy": (
                passed / total
                if total
                else 0
            ),
            "answer_accuracy": (
                answer_correct / total
                if total
                else 0
            ),
            "retrieval_accuracy": (
                retrieval_correct
                / len(retrieval_applicable)
                if retrieval_applicable
                else None
            ),
            "average_latency_ms": (
                mean(latencies)
                if latencies
                else 0
            ),
        }

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DocuRAG evaluation benchmark."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=(
            "Path to evaluation dataset JSON. "
            "Defaults to evaluation/datasets/nptel_exam.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Path for evaluation results JSON. "
            "Defaults to evaluation/latest_results.json"
        ),
    )

    parser.add_argument(
        "--api-url",
        default=BASE_API_URL,
        help=(
            "Base URL of the DocuRAG API. "
            "Defaults to http://127.0.0.1:8000"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_path = args.dataset

    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / dataset_path

    output_path = args.output

    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    if not dataset_path.exists():
        print(
            f"Dataset not found: "
            f"{dataset_path}"
        )
        return 1

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        print(
            "Dataset must contain a JSON array."
        )
        return 1

    print("=" * 78)
    print("DocuRAG Evaluation")
    print("=" * 78)
    print(
        f"Dataset: {dataset_path.name}"
    )
    print(
        f"Cases:   {len(dataset)}"
    )
    print(
        f"API:     {args.api_url}"
    )
    print("=" * 78)
    print()

    results = []

    with httpx.Client(
        timeout=120.0
    ) as client:
        for case in dataset:
            result = evaluate_case(
                client,
                case,
            )

            results.append(result)
            print_result(result)

    total = len(results)

    passed = sum(
        1
        for result in results
        if result["passed"]
    )

    failed = total - passed

    answer_correct = sum(
        1
        for result in results
        if result["answer_correct"]
    )

    retrieval_results = [
        result
        for result in results
        if result["retrieval_correct"]
        is not None
    ]

    retrieval_correct = sum(
        1
        for result in retrieval_results
        if result["retrieval_correct"]
    )

    latencies = [
        result["latency_ms"]
        for result in results
        if result["error"] is None
    ]

    average_latency = (
        mean(latencies)
        if latencies
        else 0
    )

    category_metrics = (
        calculate_category_metrics(
            results
        )
    )

    print()
    print("=" * 78)
    print("Evaluation Summary")
    print("=" * 78)

    print(
        f"Passed:              "
        f"{passed}/{total}"
    )

    print(
        f"Failed:              "
        f"{failed}/{total}"
    )

    print(
        f"Overall accuracy:    "
        f"{passed / total * 100:.2f}%"
        if total
        else "Overall accuracy:    0.00%"
    )

    print(
        f"Answer accuracy:     "
        f"{answer_correct / total * 100:.2f}%"
        if total
        else "Answer accuracy:     0.00%"
    )

    if retrieval_results:
        print(
            f"Retrieval accuracy:  "
            f"{retrieval_correct / len(retrieval_results) * 100:.2f}%"
        )
    else:
        print(
            "Retrieval accuracy:  N/A"
        )

    print(
        f"Average latency:     "
        f"{average_latency:.2f} ms"
    )

    print()
    print("Category Results")
    print("-" * 78)

    print(
        f"{'Category':18} "
        f"{'Passed':>8} "
        f"{'Accuracy':>12} "
        f"{'Answer':>12} "
        f"{'Retrieval':>12} "
        f"{'Latency':>12}"
    )

    print("-" * 78)

    for category in sorted(
        category_metrics
    ):
        metric = category_metrics[
            category
        ]

        retrieval_accuracy = metric[
            "retrieval_accuracy"
        ]

        retrieval_text = (
            "N/A"
            if retrieval_accuracy is None
            else (
                f"{retrieval_accuracy * 100:.2f}%"
            )
        )

        print(
            f"{category:18} "
            f"{metric['passed']:>3}/{metric['total']:<4} "
            f"{metric['accuracy'] * 100:>10.2f}% "
            f"{metric['answer_accuracy'] * 100:>10.2f}% "
            f"{retrieval_text:>12} "
            f"{metric['average_latency_ms']:>10.2f} ms"
        )

    print("=" * 78)

    if failed:
        print()
        print("Failed cases:")

        for result in results:
            if not result["passed"]:
                print(
                    f"- {result['id']}: "
                    f"{result['question']}"
                )

    output = {
        "dataset": dataset_path.name,
        "dataset_path": str(dataset_path),
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": (
            passed / total
            if total
            else 0
        ),
        "answer_accuracy": (
            answer_correct / total
            if total
            else 0
        ),
        "retrieval_accuracy": (
            retrieval_correct
            / len(retrieval_results)
            if retrieval_results
            else None
        ),
        "average_latency_ms": (
            average_latency
        ),
        "category_metrics": category_metrics,
        "results": results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

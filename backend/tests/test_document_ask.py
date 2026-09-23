from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.services.generation.generation_service import (
    GenerationService,
)


client = TestClient(app)


DOCUMENT_ID = "80caef9c-30fa-4494-95b8-2ab21074fb2a"


def test_ask_document_not_found() -> None:
    response = client.post(
        "/documents/00000000-0000-0000-0000-000000000000/ask",
        json={
            "query": "What is the examination date?",
            "top_k": 5,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_generation_service_refuses_empty_context() -> None:
    service = GenerationService()

    result = service.build_prompt(
        query="What is the examination date?",
        retrieved_chunks=[],
    )

    assert "No relevant document context" in result


def test_generation_service_builds_context() -> None:
    service = GenerationService()

    chunk_id = uuid4()
    document_id = uuid4()

    context = service.build_context(
        [
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "page_number": 1,
                "chunk_index": 0,
                "text": (
                    "The examination date is "
                    "20 September, 2026."
                ),
                "score": 0.8,
            }
        ]
    )

    assert "[Source 1 | Page 1]" in context
    assert "20 September, 2026" in context


def test_generation_service_builds_grounded_prompt() -> None:
    service = GenerationService()

    prompt = service.build_prompt(
        query="What is the examination date?",
        retrieved_chunks=[
            {
                "chunk_id": uuid4(),
                "document_id": uuid4(),
                "page_number": 1,
                "chunk_index": 0,
                "text": (
                    "The examination date is "
                    "20 September, 2026."
                ),
                "score": 0.8,
            }
        ],
    )

    assert "What is the examination date?" in prompt
    assert "20 September, 2026" in prompt

    assert (
        "using ONLY information explicitly stated"
        in prompt
    )

    assert (
        "Do NOT infer relationships"
        in prompt
    )


def test_generation_prompt_prevents_role_hallucination() -> None:
    """
    Regression test for the previous hallucination where Gemma
    incorrectly identified Vishwa Sabaris as an instructor.

    The document contains the person's name but does not explicitly
    state their role or identity.
    """

    service = GenerationService()

    prompt = service.build_prompt(
        query="Who is Vishwa Sabaris?",
        retrieved_chunks=[
            {
                "chunk_id": uuid4(),
                "document_id": uuid4(),
                "page_number": 1,
                "chunk_index": 0,
                "text": (
                    "NPTEL EXAM – 20 September, 2026\n\n"
                    "SEM2NOC26: CS117 Data Base Management System\n\n"
                    "Online\n\n"
                    "Vishwa Sabaris V\n\n"
                    "NOC26CS117S353802605\n\n"
                    "2605\n\n"
                    "28-07-2008\n\n"
                    "No\n\n"
                    "No\n\n"
                    "No"
                ),
                "score": 0.21,
            }
        ],
    )

    assert "Vishwa Sabaris" in prompt

    assert (
        "Do NOT infer relationships between fields merely because they appear"
        in prompt
    )

    assert (
        "next to each other."
        in prompt
    )

    assert (
        "If the requested information is completely absent"
        in prompt
    )

    assert (
        "The information is not available in the provided document."
        in prompt
    )


def test_generation_prompt_contains_document_date() -> None:
    service = GenerationService()

    prompt = service.build_prompt(
        query="What is the examination date?",
        retrieved_chunks=[
            {
                "chunk_id": uuid4(),
                "document_id": uuid4(),
                "page_number": 1,
                "chunk_index": 0,
                "text": (
                    "Sunday, 20 September, 2026"
                ),
                "score": 0.44,
            }
        ],
    )

    assert "Sunday, 20 September, 2026" in prompt
    assert "What is the examination date?" in prompt


def test_generation_prompt_contains_examination_center() -> None:
    service = GenerationService()

    prompt = service.build_prompt(
        query="What is the name of the examination center?",
        retrieved_chunks=[
            {
                "chunk_id": uuid4(),
                "document_id": uuid4(),
                "page_number": 1,
                "chunk_index": 0,
                "text": (
                    "Ranganathan Engineering College, "
                    "REC Kalvi Nagar, Viraliyur Post, "
                    "Thondamuthur"
                ),
                "score": 0.39,
            }
        ],
    )

    assert "Ranganathan Engineering College" in prompt

    assert (
        "What is the name of the examination center?"
        in prompt
    )


def test_generation_prompt_contains_refusal_instruction() -> None:
    service = GenerationService()

    prompt = service.build_prompt(
        query="What is the capital of France?",
        retrieved_chunks=[],
    )

    assert "What is the capital of France?" in prompt

    assert (
        "The information is not available "
        "in the provided document."
        in prompt
    )


def test_ask_document_with_relevant_context() -> None:
    """
    Verify that /ask returns a generated answer and
    source information when relevant chunks are retrieved.
    """

    from app.main import (
        GenerationService as MainGenerationService,
        HybridRetrievalService,
    )

    class FakeHybridRetrievalService:
        def __init__(
            self,
            *args,
            **kwargs,
        ) -> None:
            pass

        def retrieve(
            self,
            query: str,
            vector_top_k: int,
            keyword_top_k: int,
            full_text_top_k: int,
            document_id,
        ) -> list[dict]:
            return [
                {
                    "chunk_id": uuid4(),
                    "document_id": document_id,
                    "page_number": 1,
                    "chunk_index": 0,
                    "text": (
                        "The examination date is "
                        "20 September, 2026."
                    ),
                    "character_count": 43,
                    "score": 0.85,
                    "distance": 0.15,
                }
            ]

    class FakeGenerationService:
        def __init__(
            self,
            *args,
            **kwargs,
        ) -> None:
            pass

        def generate_answer(
            self,
            query: str,
            retrieved_chunks: list[dict],
        ) -> dict:
            return {
                "answer": (
                    "Sunday, 20 September, 2026"
                ),
                "sources": [
                    {
                        "chunk_id": retrieved_chunks[0][
                            "chunk_id"
                        ],
                        "document_id": retrieved_chunks[0][
                            "document_id"
                        ],
                        "page_number": 1,
                        "chunk_index": 0,
                        "score": 0.85,
                        "rerank_score": retrieved_chunks[0].get(
                            "rerank_score"
                        ),
                    }
                ],
            }

    original_retrieval_service = HybridRetrievalService
    original_generation_service = MainGenerationService

    try:
        import app.main as main_module

        main_module.HybridRetrievalService = (
            FakeHybridRetrievalService
        )

        main_module.GenerationService = (
            FakeGenerationService
        )

        response = client.post(
            f"/documents/{DOCUMENT_ID}/ask",
            json={
                "query": "What is the examination date?",
                "top_k": 5,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert (
            data["answer"]
            == "Sunday, 20 September, 2026"
        )

        assert len(data["sources"]) == 1

        assert (
            data["sources"][0]["page_number"]
            == 1
        )

        assert (
            data["sources"][0]["score"]
            == 0.85
        )

    finally:
        main_module.HybridRetrievalService = (
            original_retrieval_service
        )

        main_module.GenerationService = (
            original_generation_service
        )


def test_ask_document_without_relevant_context_does_not_call_llm() -> None:
    """
    Verify that /ask returns the safe refusal when retrieval
    returns no relevant chunks and that the LLM is never called.
    """

    from app.main import (
        GenerationService as MainGenerationService,
        HybridRetrievalService,
    )

    class FakeHybridRetrievalService:
        def __init__(
            self,
            *args,
            **kwargs,
        ) -> None:
            pass

        def retrieve(
            self,
            query: str,
            vector_top_k: int,
            keyword_top_k: int,
            full_text_top_k: int,
            document_id,
        ) -> list[dict]:
            return []

    class FakeGenerationService:
        llm_called = False

        def __init__(
            self,
            *args,
            **kwargs,
        ) -> None:
            pass

        def generate_answer(
            self,
            query: str,
            retrieved_chunks: list[dict],
        ) -> dict:
            FakeGenerationService.llm_called = True

            return {
                "answer": "This should never be returned.",
                "sources": [],
            }

    original_retrieval_service = HybridRetrievalService
    original_generation_service = MainGenerationService

    try:
        import app.main as main_module

        main_module.HybridRetrievalService = (
            FakeHybridRetrievalService
        )

        main_module.GenerationService = (
            FakeGenerationService
        )

        response = client.post(
            f"/documents/{DOCUMENT_ID}/ask",
            json={
                "query": "What is the capital of France?",
                "top_k": 5,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert (
            data["answer"]
            == (
                "The information is not available "
                "in the provided document."
            )
        )

        assert data["sources"] == []

        assert (
            FakeGenerationService.llm_called
            is False
        )

    finally:
        main_module.HybridRetrievalService = (
            original_retrieval_service
        )

        main_module.GenerationService = (
            original_generation_service
        )

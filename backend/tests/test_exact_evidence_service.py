from app.services.exact_evidence.exact_evidence_service import (
    ExactEvidenceService,
)


def test_identifier_is_detected():
    query = "NOC26CS117S353802605"

    assert (
        ExactEvidenceService.is_literal_identifier(query)
        is True
    )


def test_natural_language_question_is_not_identifier():
    query = "What is NOC26CS117S353802605?"

    assert (
        ExactEvidenceService.is_literal_identifier(query)
        is False
    )


def test_identifier_with_spaces_is_not_identifier():
    query = "NOC26CS117 S353802605"

    assert (
        ExactEvidenceService.is_literal_identifier(query)
        is False
    )


def test_identifier_must_contain_letters_and_digits():
    assert (
        ExactEvidenceService.is_literal_identifier(
            "123456789"
        )
        is False
    )

    assert (
        ExactEvidenceService.is_literal_identifier(
            "ABCDEFGH"
        )
        is False
    )


def test_exact_identifier_is_found_in_chunk():
    chunks = [
        {
            "chunk_id": "chunk-1",
            "page_number": 1,
            "text": (
                "Vishwa Sabaris V\n"
                "NOC26CS117S353802605\n"
                "2605"
            ),
        }
    ]

    result = ExactEvidenceService.find_exact_identifier(
        "NOC26CS117S353802605",
        chunks,
    )

    assert result is not None
    assert result["value"] == "NOC26CS117S353802605"
    assert result["chunk"]["chunk_id"] == "chunk-1"


def test_identifier_not_in_evidence_returns_none():
    chunks = [
        {
            "chunk_id": "chunk-1",
            "page_number": 1,
            "text": "Vishwa Sabaris V\n2605",
        }
    ]

    result = ExactEvidenceService.find_exact_identifier(
        "NOC26CS117S353802605",
        chunks,
    )

    assert result is None


def test_meaning_question_does_not_trigger_exact_extraction():
    chunks = [
        {
            "chunk_id": "chunk-1",
            "page_number": 1,
            "text": (
                "Vishwa Sabaris V\n"
                "NOC26CS117S353802605\n"
                "2605"
            ),
        }
    ]

    result = ExactEvidenceService.find_exact_identifier(
        "What does NOC26CS117S353802605 represent?",
        chunks,
    )

    assert result is None


def test_meaning_question_identifier_is_detected():
    chunks = [
        {
            "chunk_id": "chunk-1",
            "page_number": 1,
            "text": (
                "Vishwa Sabaris V\n"
                "NOC26CS117S353802605\n"
                "2605"
            ),
        }
    ]

    result = (
        ExactEvidenceService.find_identifier_in_meaning_question(
            "What does NOC26CS117S353802605 represent?",
            chunks,
        )
    )

    assert result is not None
    assert (
        result["identifier"]
        == "NOC26CS117S353802605"
    )
    assert result["chunk"]["chunk_id"] == "chunk-1"


def test_non_meaning_question_returns_none():
    chunks = [
        {
            "chunk_id": "chunk-1",
            "page_number": 1,
            "text": (
                "Vishwa Sabaris V\n"
                "NOC26CS117S353802605\n"
                "2605"
            ),
        }
    ]

    result = (
        ExactEvidenceService.find_identifier_in_meaning_question(
            "What is the examination date?",
            chunks,
        )
    )

    assert result is None

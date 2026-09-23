import pytest

from backend.app.services.answerability.answerability_service import (
    AnswerabilityService,
)


HALL_TICKET_TEXT = """
NPTEL EXAM – 20 September, 2026
SEM2NOC26: CS117 Data Base Management System 20 Sep - Online
Vishwa Sabaris V
NOC26CS117S353802605 2605
28-07-2008
No No No
Sunday, 20 September, 2026
iON Digital Zone iDZ Thondamuthur
Ranganathan Engineering College, REC Kalvi Nagar,
Viraliyur Post, Thondamuthur via Coimbatore,
Tamil Nadu, India 641109.
""".strip()


def make_chunk(
    text: str,
    rerank_score: float = 0.0,
) -> dict:
    return {
        "text": text,
        "rerank_score": rerank_score,
        "page_number": 1,
        "chunk_index": 0,
    }


def test_unsupported_family_relationship_is_rejected():
    service = AnswerabilityService()

    result = service.check(
        query="What is the candidate's father's name?",
        reranked_chunks=[
            make_chunk(HALL_TICKET_TEXT),
        ],
    )

    assert result["answerable"] is False
    assert result["reason"] == "unsupported_relationship"


def test_unsupported_phone_number_is_rejected():
    service = AnswerabilityService()

    result = service.check(
        query="What is the candidate's phone number?",
        reranked_chunks=[
            make_chunk(HALL_TICKET_TEXT),
        ],
    )

    assert result["answerable"] is False
    assert result["reason"] == "unsupported_relationship"


def test_unsupported_student_role_is_rejected():
    service = AnswerabilityService()

    result = service.check(
        query="Is Vishwa Sabaris V a student?",
        reranked_chunks=[
            make_chunk(HALL_TICKET_TEXT),
        ],
    )

    assert result["answerable"] is False
    assert result["reason"] == "unsupported_relationship"


def test_unsupported_instructor_role_is_rejected():
    service = AnswerabilityService()

    result = service.check(
        query="Is Vishwa Sabaris V an instructor?",
        reranked_chunks=[
            make_chunk(HALL_TICKET_TEXT),
        ],
    )

    assert result["answerable"] is False
    assert result["reason"] == "unsupported_relationship"


def test_identifier_meaning_without_explicit_definition_is_rejected():
    service = AnswerabilityService()

    result = service.check(
        query=(
            "Does 2605 represent the candidate's "
            "registration number?"
        ),
        reranked_chunks=[
            make_chunk(HALL_TICKET_TEXT),
        ],
    )

    assert result["answerable"] is False
    assert result["reason"] == "unsupported_relationship"


def test_explicit_student_role_is_supported():
    service = AnswerabilityService()

    result = service.check(
        query="Is Vishwa Sabaris V a student?",
        reranked_chunks=[
            make_chunk(
                "Vishwa Sabaris V is a student.",
            ),
        ],
    )

    assert result["answerable"] is True


def test_date_of_birth_extraction_remains_answerable():
    service = AnswerabilityService()

    result = service.check(
        query="What is the candidate's date of birth?",
        reranked_chunks=[
            make_chunk(HALL_TICKET_TEXT),
        ],
    )

    assert result["answerable"] is True


def test_empty_retrieval_is_not_answerable():
    service = AnswerabilityService()

    result = service.check(
        query="What is the examination date?",
        reranked_chunks=[],
    )

    assert result["answerable"] is False


def test_empty_text_is_not_answerable():
    service = AnswerabilityService()

    result = service.check(
        query="What is the examination date?",
        reranked_chunks=[
            make_chunk(""),
        ],
    )

    assert result["answerable"] is False


def test_exact_query_match_is_answerable():
    service = AnswerabilityService()

    result = service.check(
        query="NOC26CS117S353802605",
        reranked_chunks=[
            make_chunk(HALL_TICKET_TEXT),
        ],
    )

    assert result["answerable"] is True


def test_meaningful_overlap_is_answerable():
    service = AnswerabilityService()

    result = service.check(
        query="What is the examination date?",
        reranked_chunks=[
            make_chunk(HALL_TICKET_TEXT),
        ],
    )

    assert result["answerable"] is True


def test_low_reranker_score_does_not_reject_valid_evidence():
    service = AnswerabilityService()

    result = service.check(
        query="What is the examination center?",
        reranked_chunks=[
            make_chunk(
                HALL_TICKET_TEXT,
                rerank_score=-10.228618621826172,
            ),
        ],
    )

    assert result["answerable"] is True


def test_weak_lexical_overlap_does_not_reject_semantic_evidence():
    service = AnswerabilityService()

    result = service.check(
        query="Where should the candidate report?",
        reranked_chunks=[
            make_chunk(
                "iON Digital Zone iDZ Thondamuthur "
                "Ranganathan Engineering College."
            ),
        ],
    )

    assert result["answerable"] is True


def test_stopwords_do_not_inflate_overlap():
    service = AnswerabilityService()

    result = service.check(
        query="What is the candidate's the the the name?",
        reranked_chunks=[
            make_chunk("Vishwa Sabaris V"),
        ],
    )

    assert result["answerable"] is True


def test_semantic_date_question_is_answerable():
    service = AnswerabilityService()

    result = service.check(
        query="When is the test scheduled?",
        reranked_chunks=[
            make_chunk(HALL_TICKET_TEXT),
        ],
    )

    assert result["answerable"] is True


def test_name_without_role_is_not_automatically_rejected():
    service = AnswerabilityService()

    result = service.check(
        query="Who is the candidate?",
        reranked_chunks=[
            make_chunk(HALL_TICKET_TEXT),
        ],
    )

    assert result["answerable"] is True


def test_empty_query_is_rejected():
    service = AnswerabilityService()

    with pytest.raises(ValueError):
        service.check(
            query="",
            reranked_chunks=[
                make_chunk(HALL_TICKET_TEXT),
            ],
        )


def test_invalid_positive_token_overlap_threshold_is_rejected():
    with pytest.raises(ValueError):
        AnswerabilityService(
            min_token_overlap=-0.1,
        )


def test_invalid_negative_token_overlap_threshold_is_rejected():
    with pytest.raises(ValueError):
        AnswerabilityService(
            min_token_overlap=1.1,
        )


def test_invalid_positive_evidence_density_threshold_is_rejected():
    with pytest.raises(ValueError):
        AnswerabilityService(
            min_evidence_density=-0.1,
        )


def test_invalid_negative_evidence_density_threshold_is_rejected():
    with pytest.raises(ValueError):
        AnswerabilityService(
            min_evidence_density=1.1,
        )

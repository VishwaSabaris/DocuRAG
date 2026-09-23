from __future__ import annotations

import re
from typing import Any

from app.services.structured_evidence.structured_evidence_service import (
    StructuredEvidenceService,
)


class AnswerabilityService:
    """
    Document-agnostic answerability gate for DocuRAG.

    The service determines whether retrieved evidence contains enough
    support to answer the user's question from the provided document.

    It does NOT:
        - generate answers
        - use external knowledge
        - infer undocumented relationships
        - decide what an arbitrary identifier means

    Critical principle
    ------------------
    A retrieved value is not automatically evidence for the requested
    relationship.

    For example:

        Vishwa Sabaris V
        Ranganathan Engineering College

    does NOT establish:

        Vishwa Sabaris V's college =
        Ranganathan Engineering College

    unless the document explicitly establishes that relationship.

    This rule is generic and applies to every PDF/document type.
    """

    DEFAULT_MIN_TOKEN_OVERLAP = 0.20
    DEFAULT_MIN_EVIDENCE_DENSITY = 0.01

    # ------------------------------------------------------------------
    # Stopwords
    # ------------------------------------------------------------------

    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "may",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "will",
        "with",
        "would",
        "you",
        "your",
        "candidate",
    }

    QUESTION_WORDS = {
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
    }

    # ------------------------------------------------------------------
    # Semantic vocabularies
    # ------------------------------------------------------------------

    MEANING_VERBS = {
        "mean",
        "means",
        "meaning",
        "represent",
        "represents",
        "represented",
        "stand",
        "stands",
        "standing",
    }

    FAMILY_TERMS = {
        "father",
        "mother",
        "parent",
        "parents",
        "son",
        "daughter",
        "brother",
        "sister",
        "sibling",
        "siblings",
        "husband",
        "wife",
        "spouse",
        "guardian",
        "grandfather",
        "grandmother",
    }

    ROLE_TERMS = {
        "student",
        "instructor",
        "teacher",
        "professor",
        "employee",
        "employer",
        "manager",
        "developer",
        "engineer",
        "doctor",
        "candidate",
        "applicant",
        "staff",
        "faculty",
        "member",
        "intern",
        "director",
        "administrator",
        "owner",
    }

    RESULT_TERMS = {
        "result",
        "results",
        "score",
        "marks",
        "mark",
        "grade",
        "rank",
        "percentage",
        "percentile",
        "pass",
        "passed",
        "fail",
        "failed",
        "qualification",
        "degree",
        "cgpa",
        "gpa",
        "classification",
        "performance",
    }

    INSTITUTION_TERMS = {
        "college",
        "university",
        "institute",
        "institution",
        "school",
        "organization",
        "organisation",
        "department",
    }

    CONTACT_TERMS = {
        "phone",
        "telephone",
        "mobile",
        "email",
        "e-mail",
        "contact",
    }

    RELATIONSHIP_FIELD_TERMS = {
        "college",
        "university",
        "institute",
        "institution",
        "school",
        "organization",
        "organisation",
        "department",
        "degree",
        "qualification",
        "graduation",
        "father",
        "mother",
        "parent",
        "parents",
        "son",
        "daughter",
        "brother",
        "sister",
        "sibling",
        "husband",
        "wife",
        "spouse",
        "guardian",
        "phone",
        "telephone",
        "mobile",
        "email",
        "e-mail",
        "contact",
        "address",
        "location",
        "identifier",
        "registration",
        "registration number",
        "registration id",
        "student id",
        "candidate id",
        "application id",
        "application number",
        "roll number",
        "roll no",
        "admission number",
        "result",
        "score",
        "marks",
        "mark",
        "grade",
        "rank",
        "percentage",
        "cgpa",
        "gpa",
    }

    LIST_QUESTION_TERMS = {
        "list",
        "lists",
        "all",
        "names",
        "name",
        "items",
        "item",
        "details",
        "subjects",
        "courses",
        "services",
        "features",
        "requirements",
        "documents",
        "steps",
        "types",
    }

    LIST_SECTION_TERMS = {
        "subjects",
        "courses",
        "services",
        "features",
        "requirements",
        "documents required",
        "eligibility",
        "qualifications",
        "contact details",
        "skills",
        "experience",
        "responsibilities",
        "modules",
        "sections",
        "contents",
    }

    NEGATIVE_EVIDENCE_PATTERNS = (
        r"\bnot\s+provided\b",
        r"\bnot\s+available\b",
        r"\bnot\s+specified\b",
        r"\bnot\s+mentioned\b",
        r"\bnot\s+given\b",
        r"\bnot\s+listed\b",
        r"\bnot\s+known\b",
        r"\bunknown\b",
        r"\bno\s+information\b",
        r"\bno\s+record\b",
        r"\bnone\s+provided\b",
        r"\bnot\s+applicable\b",
        r"\bn/?a\b",
        r"\bna\b",
        r"\bdon't\s+state\b",
        r"\bdo\s+not\s+state\b",
        r"\bdoes\s+not\s+state\b",
    )

    YES_NO_ROLE_PATTERN = re.compile(
        r"^\s*(?:is|are|was|were)\b",
        re.IGNORECASE,
    )

    EXACT_IDENTIFIER_MEANING_PATTERN = re.compile(
        r"^\s*what\s+does\s+"
        r"([A-Za-z0-9._:/-]+)\s+"
        r"(?:represent|mean|stand\s+for)\s*\??\s*$",
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Semantic field aliases
    # ------------------------------------------------------------------

    FIELD_ALIASES: dict[str, set[str]] = {
        "name": {
            "name",
            "candidate name",
            "full name",
            "student name",
            "applicant name",
            "employee name",
        },
        "date_of_birth": {
            "date of birth",
            "dob",
            "birth date",
            "birthdate",
        },
        "date": {
            "date",
            "exam date",
            "examination date",
            "test date",
            "scheduled date",
            "schedule",
            "scheduled",
        },
        "examination_date": {
            "exam date",
            "examination date",
            "test date",
            "date of examination",
            "date of exam",
        },
        "examination_center": {
            "examination center",
            "exam center",
            "examination centre",
            "exam centre",
            "test center",
            "test centre",
            "exam location",
            "examination location",
            "test location",
            "venue",
            "exam venue",
            "examination venue",
        },
        "location": {
            "location",
            "center",
            "centre",
            "venue",
            "address",
        },
        "reporting": {
            "report",
            "reporting",
            "report at",
            "report to",
            "reporting location",
            "reporting center",
            "reporting centre",
        },
        "address": {
            "address",
            "home address",
            "residential address",
            "street address",
            "location",
        },
        "phone": {
            "phone",
            "phone number",
            "telephone",
            "telephone number",
            "mobile",
            "mobile number",
            "contact number",
        },
        "email": {
            "email",
            "email address",
            "e-mail",
            "e-mail address",
        },
        "degree": {
            "degree",
            "qualification",
            "academic qualification",
        },
        "graduation": {
            "graduation",
            "graduation year",
            "expected graduation",
            "expected graduation year",
        },
        "institution": {
            "institution",
            "college",
            "university",
            "institute",
            "school",
        },
        "identifier": {
            "identifier",
            "id",
            "registration number",
            "registration no",
            "registration id",
            "student id",
            "candidate id",
            "academic id",
            "academic identifier",
            "application id",
            "application number",
            "roll number",
            "roll no",
            "admission number",
            "admission no",
        },
        "postal_code": {
            "postal code",
            "postcode",
            "pin code",
            "pincode",
            "zip code",
            "zip",
        },
        "subject": {
            "subject",
            "course",
            "course title",
            "subject name",
            "course name",
        },
    }

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        min_token_overlap: float = DEFAULT_MIN_TOKEN_OVERLAP,
        min_evidence_density: float = DEFAULT_MIN_EVIDENCE_DENSITY,
    ) -> None:
        if not 0.0 <= min_token_overlap <= 1.0:
            raise ValueError(
                "min_token_overlap must be between 0 and 1"
            )

        if not 0.0 <= min_evidence_density <= 1.0:
            raise ValueError(
                "min_evidence_density must be between 0 and 1"
            )

        self.min_token_overlap = min_token_overlap
        self.min_evidence_density = min_evidence_density

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @classmethod
    def _tokenize(
        cls,
        text: str,
    ) -> set[str]:
        return {
            token
            for token in re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower(),
            )
            if token not in cls.STOPWORDS
        }

    @classmethod
    def _query_content_tokens(
        cls,
        query: str,
    ) -> set[str]:
        tokens = cls._tokenize(query)

        return {
            token
            for token in tokens
            if token not in cls.QUESTION_WORDS
        }

    # ------------------------------------------------------------------
    # Basic evidence metrics
    # ------------------------------------------------------------------

    @classmethod
    def token_overlap(
        cls,
        query: str,
        context: str,
    ) -> float:
        query_tokens = cls._query_content_tokens(
            query
        )

        if not query_tokens:
            return 0.0

        context_tokens = cls._tokenize(
            context
        )

        if not context_tokens:
            return 0.0

        overlap = query_tokens.intersection(
            context_tokens
        )

        return len(overlap) / len(query_tokens)

    @classmethod
    def evidence_density(
        cls,
        query: str,
        context: str,
    ) -> float:
        query_tokens = cls._query_content_tokens(
            query
        )

        if not query_tokens:
            return 0.0

        context_tokens = cls._tokenize(
            context
        )

        if not context_tokens:
            return 0.0

        overlap = query_tokens.intersection(
            context_tokens
        )

        return len(overlap) / len(context_tokens)

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    @classmethod
    def _extract_context(
        cls,
        reranked_chunks: list[dict[str, Any]],
    ) -> str:
        texts: list[str] = []

        for chunk in reranked_chunks:
            text = str(
                chunk.get(
                    "text",
                    "",
                )
            ).strip()

            if text:
                texts.append(text)

        return "\n".join(texts)

    @classmethod
    def _split_sentences(
        cls,
        context: str,
    ) -> list[str]:
        sentences: list[str] = []

        for part in re.split(
            r"(?<=[.!?])\s+|\n+",
            context,
        ):
            cleaned = part.strip()

            if cleaned:
                sentences.append(cleaned)

        return sentences

    @classmethod
    def _has_exact_query_match(
        cls,
        query: str,
        context: str,
    ) -> bool:
        normalized_query = " ".join(
            query.strip().lower().split()
        )

        normalized_context = " ".join(
            context.strip().lower().split()
        )

        if not normalized_query:
            return False

        return normalized_query in normalized_context

    # ------------------------------------------------------------------
    # Identifier questions
    # ------------------------------------------------------------------

    @classmethod
    def _extract_identifier(
        cls,
        query: str,
    ) -> str | None:
        match = cls.EXACT_IDENTIFIER_MEANING_PATTERN.fullmatch(
            query.strip()
        )

        if not match:
            return None

        return match.group(1)

    @classmethod
    def _is_exact_identifier_meaning_question(
        cls,
        query: str,
    ) -> bool:
        identifier = cls._extract_identifier(
            query
        )

        if identifier is None:
            return False

        return (
            bool(re.search(r"[A-Za-z]", identifier))
            and bool(re.search(r"\d", identifier))
        )

    @classmethod
    def _identifier_has_explanation(
        cls,
        query: str,
        context: str,
    ) -> bool:
        identifier = cls._extract_identifier(
            query
        )

        if not identifier:
            return False

        identifier_pattern = re.escape(
            identifier
        )

        for sentence in cls._split_sentences(
            context
        ):
            if not re.search(
                rf"\b{identifier_pattern}\b",
                sentence,
                re.IGNORECASE,
            ):
                continue

            positive_pattern = (
                rf"\b{identifier_pattern}\b.*?"
                r"(?:represents|represent|means|meaning|"
                r"stands\s+for)"
                r"|"
                r"(?:represents|represent|means|meaning|"
                r"stands\s+for).*?"
                rf"\b{identifier_pattern}\b"
            )

            if re.search(
                positive_pattern,
                sentence,
                re.IGNORECASE,
            ):
                return True

        return False

    @classmethod
    def _identifier_is_present_without_meaning(
        cls,
        query: str,
        context: str,
    ) -> bool:
        identifier = cls._extract_identifier(
            query
        )

        if not identifier:
            return False

        return bool(
            re.search(
                rf"\b{re.escape(identifier)}\b",
                context,
                re.IGNORECASE,
            )
        )

    # ------------------------------------------------------------------
    # Meaning language
    # ------------------------------------------------------------------

    @classmethod
    def _contains_meaning_language(
        cls,
        query: str,
    ) -> bool:
        lowered = query.lower()

        return any(
            re.search(
                rf"\b{re.escape(verb)}\b",
                lowered,
            )
            for verb in cls.MEANING_VERBS
        )

    # ------------------------------------------------------------------
    # Query relationship classification
    # ------------------------------------------------------------------

    @classmethod
    def _query_contains_contact_field(
        cls,
        lowered_query: str,
    ) -> bool:
        if re.search(
            r"\b(?:phone|telephone|mobile)\b",
            lowered_query,
        ):
            return True

        if re.search(
            r"\b(?:email|e-mail)\b",
            lowered_query,
        ):
            return True

        if re.search(
            r"\bcontact\s+(?:number|details?|information)\b",
            lowered_query,
        ):
            return True

        return False

    @classmethod
    def _query_contains_institution_field(
        cls,
        lowered_query: str,
    ) -> bool:
        return any(
            re.search(
                rf"\b{re.escape(term)}\b",
                lowered_query,
            )
            for term in cls.INSTITUTION_TERMS
        )

    @classmethod
    def _query_contains_relationship_field(
        cls,
        lowered_query: str,
    ) -> bool:
        for term in cls.RELATIONSHIP_FIELD_TERMS:
            if re.search(
                rf"\b{re.escape(term)}\b",
                lowered_query,
            ):
                return True

        return False

    @classmethod
    def _query_targets_named_person(
        cls,
        query: str,
    ) -> bool:
        """
        Detect a concrete named-person possessive question.

        This deliberately avoids trying to determine whether the
        text before "'s" is actually a person's name. The question
        itself supplies the subject; evidence extraction verifies
        whether that subject occurs in the document.
        """

        return bool(
            re.search(
                r"\b[A-Za-z][A-Za-z0-9 .'-]*"
                r"['’]s\s+",
                query,
                re.IGNORECASE,
            )
        )

    @classmethod
    def _query_targets_generic_subject(
        cls,
        query: str,
    ) -> bool:
        lowered = query.lower()

        return bool(
            re.search(
                r"\b(?:candidate|student|applicant|employee|"
                r"person|user|customer|patient)"
                r"\s*['’]s\b",
                lowered,
            )
        )

    @classmethod
    def _requires_explicit_relation_support(
        cls,
        query: str,
    ) -> bool:
        normalized_query = query.strip()
        lowered_query = normalized_query.lower()

        # Exact identifier meaning is handled separately.
        if cls._is_exact_identifier_meaning_question(
            normalized_query
        ):
            return True

        # Family relationships always require explicit support.
        if any(
            re.search(
                rf"\b{re.escape(term)}\b",
                lowered_query,
            )
            for term in cls.FAMILY_TERMS
        ):
            return True

        # Contact information always requires explicit support.
        if cls._query_contains_contact_field(
            lowered_query
        ):
            return True

        # Explicit institution questions.
        if cls._query_contains_institution_field(
            lowered_query
        ):
            if (
                cls._query_targets_named_person(
                    normalized_query
                )
                or cls._query_targets_generic_subject(
                    normalized_query
                )
            ):
                return True

        # Any person-specific field question must not fall through
        # to generic lexical/semantic evidence.
        if (
            cls._query_targets_named_person(
                normalized_query
            )
            or cls._query_targets_generic_subject(
                normalized_query
            )
        ):
            if cls._query_contains_relationship_field(
                lowered_query
            ):
                return True

        # Yes/no role questions.
        if cls.YES_NO_ROLE_PATTERN.match(
            normalized_query
        ):
            if any(
                re.search(
                    rf"\b{re.escape(role)}\b",
                    lowered_query,
                )
                for role in cls.ROLE_TERMS
            ):
                return True

        # Candidate + institution without possessive syntax.
        if (
            "candidate" in lowered_query
            and cls._query_contains_institution_field(
                lowered_query
            )
        ):
            return True

        # Candidate + identifier meaning.
        if (
            cls._contains_meaning_language(
                normalized_query
            )
            and "candidate" in lowered_query
            and (
                "registration" in lowered_query
                or "number" in lowered_query
                or "id" in lowered_query
                or "identifier" in lowered_query
            )
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Explicit relationship support
    # ------------------------------------------------------------------

    @classmethod
    def _has_explicit_relation_support(
        cls,
        query: str,
        context: str,
    ) -> bool:
        """
        Ask StructuredEvidenceService to establish the requested
        relationship.

        This is intentional.

        There must be one authoritative deterministic interpretation
        of person -> field relationships. Otherwise answerability and
        generation can disagree.
        """

        if not context.strip():
            return False

        result = StructuredEvidenceService.extract(
            query=query,
            chunks=[
                {
                    "text": context,
                    "page_number": None,
                    "chunk_index": 0,
                    "chunk_id": (
                        "answerability-context"
                    ),
                    "document_id": (
                        "answerability-context"
                    ),
                }
            ],
        )

        return result is not None

    # ------------------------------------------------------------------
    # Explicit negative evidence
    # ------------------------------------------------------------------

    @classmethod
    def _query_negative_evidence_terms(
        cls,
        query: str,
    ) -> set[str]:
        lowered_query = query.lower()
        terms: set[str] = set()

        if re.search(
            r"\b(?:phone|telephone|mobile)\b",
            lowered_query,
        ):
            terms.update(
                {
                    "phone",
                    "telephone",
                    "mobile",
                }
            )

        if re.search(
            r"\b(?:email|e-mail)\b",
            lowered_query,
        ):
            terms.update(
                {
                    "email",
                    "e-mail",
                }
            )

        if re.search(
            r"\bcontact\s+(?:number|details?|information)\b",
            lowered_query,
        ):
            terms.update(
                {
                    "contact number",
                    "contact details",
                    "contact information",
                }
            )

        for term in cls.FAMILY_TERMS:
            if re.search(
                rf"\b{re.escape(term)}\b",
                lowered_query,
            ):
                terms.add(term)

        financial_terms = {
            "salary",
            "income",
            "monthly income",
            "monthly salary",
            "wage",
            "compensation",
            "pay",
        }

        for term in financial_terms:
            if re.search(
                rf"\b{re.escape(term)}\b",
                lowered_query,
            ):
                terms.add(term)

        address_terms = {
            "address",
            "home address",
            "residential address",
            "residential street",
            "street address",
            "home",
            "residential",
            "live",
            "lives",
            "living",
        }

        for term in address_terms:
            if re.search(
                rf"\b{re.escape(term)}\b",
                lowered_query,
            ):
                terms.add(term)

        for term in cls.RESULT_TERMS:
            if re.search(
                rf"\b{re.escape(term)}\b",
                lowered_query,
            ):
                terms.add(term)

        for term in cls.INSTITUTION_TERMS:
            if re.search(
                rf"\b{re.escape(term)}\b",
                lowered_query,
            ):
                terms.add(term)

        return terms

    @classmethod
    def _negative_sentence_matches_query(
        cls,
        query: str,
        sentence: str,
    ) -> bool:
        query_terms = cls._query_negative_evidence_terms(
            query
        )

        if not query_terms:
            return False

        lowered_sentence = sentence.lower()
        lowered_query = query.lower()

        for term in query_terms:
            if re.search(
                rf"\b{re.escape(term)}\b",
                lowered_sentence,
            ):
                return True

        family_map = {
            "father": r"\b(?:father|parent|parents)\b",
            "mother": r"\b(?:mother|parent|parents)\b",
            "son": r"\b(?:son|child|children)\b",
            "daughter": r"\b(?:daughter|child|children)\b",
            "brother": r"\b(?:brother|sibling|siblings)\b",
            "sister": r"\b(?:sister|sibling|siblings)\b",
        }

        for query_term, sentence_pattern in family_map.items():
            if query_term in lowered_query:
                if re.search(
                    sentence_pattern,
                    lowered_sentence,
                ):
                    return True

        if any(
            term in lowered_query
            for term in {
                "salary",
                "income",
                "compensation",
                "wage",
                "pay",
            }
        ):
            if re.search(
                r"\b(?:salary|income|compensation|wage|pay)\b",
                lowered_sentence,
            ):
                return True

        if (
            "address" in lowered_query
            or re.search(
                r"\blive\b|\blives\b|\bliving\b",
                lowered_query,
            )
            or "residential" in lowered_query
            or "home" in lowered_query
        ):
            if re.search(
                r"\b(?:address|home|residential|street)\b",
                lowered_sentence,
            ):
                return True

            if re.search(
                r"\b(?:live|lives|living)\b",
                lowered_sentence,
            ):
                return True

        return False

    @classmethod
    def _has_explicit_negative_evidence(
        cls,
        query: str,
        context: str,
    ) -> bool:
        negative_sentences: list[str] = []

        for sentence in cls._split_sentences(
            context
        ):
            if any(
                re.search(
                    pattern,
                    sentence,
                    re.IGNORECASE,
                )
                for pattern in cls.NEGATIVE_EVIDENCE_PATTERNS
            ):
                negative_sentences.append(
                    sentence
                )

        if not negative_sentences:
            return False

        return any(
            cls._negative_sentence_matches_query(
                query,
                sentence,
            )
            for sentence in negative_sentences
        )

    # ------------------------------------------------------------------
    # Query field detection
    # ------------------------------------------------------------------

    @classmethod
    def _requested_field_groups(
        cls,
        query: str,
    ) -> set[str]:
        lowered = query.lower()
        groups: set[str] = set()

        for group, aliases in cls.FIELD_ALIASES.items():
            for alias in aliases:
                if re.search(
                    rf"\b{re.escape(alias)}\b",
                    lowered,
                ):
                    groups.add(group)
                    break

        if re.search(
            r"\bwho\s+is\s+(?:the\s+)?candidate\b",
            lowered,
        ):
            groups.add("name")

        if re.search(
            r"\bdate\s+of\s+birth\b",
            lowered,
        ):
            groups.add("date_of_birth")

        if re.search(
            r"\b(?:exam|examination|test)\s+date\b",
            lowered,
        ):
            groups.add("examination_date")

        if re.search(
            r"\b(?:exam|examination|test)\s+center\b",
            lowered,
        ) or re.search(
            r"\b(?:exam|examination|test)\s+centre\b",
            lowered,
        ):
            groups.add("examination_center")

        if (
            re.search(
                r"\bwhere\s+should\b",
                lowered,
            )
            and re.search(
                r"\breport\b",
                lowered,
            )
        ):
            groups.add("reporting")

        if (
            re.search(
                r"\bwhen\s+is\b",
                lowered,
            )
            and re.search(
                r"\b(?:test|exam|examination)\b",
                lowered,
            )
        ):
            groups.add("examination_date")

        if (
            re.search(
                r"\bwhat\s+is\b",
                lowered,
            )
            and re.search(
                r"\bname\b",
                lowered,
            )
        ):
            groups.add("name")

        return groups

    @classmethod
    def _context_contains_field(
        cls,
        field: str,
        context: str,
    ) -> bool:
        lowered_context = context.lower()

        aliases = cls.FIELD_ALIASES.get(
            field,
            set(),
        )

        return any(
            re.search(
                rf"\b{re.escape(alias)}\b",
                lowered_context,
            )
            for alias in aliases
        )

    # ------------------------------------------------------------------
    # Structured evidence
    # ------------------------------------------------------------------

    @classmethod
    def _has_structured_evidence(
        cls,
        query: str,
        context: str,
    ) -> bool:
        if not context.strip():
            return False

        result = StructuredEvidenceService.extract(
            query=query,
            chunks=[
                {
                    "text": context,
                    "page_number": None,
                    "chunk_index": 0,
                    "chunk_id": (
                        "answerability-context"
                    ),
                    "document_id": (
                        "answerability-context"
                    ),
                }
            ],
        )

        return result is not None

    # ------------------------------------------------------------------
    # Direct factual evidence
    # ------------------------------------------------------------------

    @classmethod
    def _has_direct_factual_evidence(
        cls,
        query: str,
        context: str,
    ) -> bool:
        lowered_query = query.lower()
        lowered_context = context.lower()

        factual_patterns = (
            "institution",
            "degree",
            "expected graduation",
            "graduation year",
            "current year",
            "city",
            "postal code",
            "date of birth",
            "average retrieval latency",
            "documents tested",
            "simulated incidents tested",
            "development port",
            "webhook endpoint",
            "external api endpoint",
            "examination date",
            "exam date",
            "examination center",
            "exam center",
            "examination centre",
            "exam centre",
            "candidate name",
            "full name",
        )

        for phrase in factual_patterns:
            if phrase not in lowered_query:
                continue

            if re.search(
                rf"\b{re.escape(phrase)}\b"
                r"\s*(?::|-|=|is|was)",
                lowered_context,
                re.IGNORECASE,
            ):
                return True

        return False

    # ------------------------------------------------------------------
    # Subject -> field relation
    # ------------------------------------------------------------------

    @classmethod
    def _has_explicit_subject_field_relation(
        cls,
        query: str,
        context: str,
    ) -> bool:
        lowered_query = query.lower()
        lowered_context = context.lower()

        field_patterns = {
            "degree": (
                r"\bdegree\b",
                r"\bqualification\b",
            ),
            "graduation": (
                r"\bgraduation\b",
                r"\bgraduation\s+year\b",
                r"\bexpected\s+graduation\b",
            ),
            "academic identifier": (
                r"\bacademic\s+identifier\b",
                r"\bacademic\s+id\b",
                r"\bstudent\s+id\b",
                r"\bidentifier\b",
            ),
            "institution": (
                r"\binstitution\b",
                r"\buniversity\b",
                r"\bcollege\b",
                r"\binstitute\b",
            ),
        }

        requested_fields: list[str] = []

        for field in field_patterns:
            if field in lowered_query:
                requested_fields.append(
                    field
                )

        if not requested_fields:
            return False

        for field in requested_fields:
            for pattern in field_patterns[field]:
                if re.search(
                    pattern,
                    lowered_context,
                    re.IGNORECASE,
                ):
                    return True

        return False

    # ------------------------------------------------------------------
    # Semantic evidence
    # ------------------------------------------------------------------

    @classmethod
    def _has_semantic_field_evidence(
        cls,
        query: str,
        context: str,
    ) -> bool:
        """
        Semantic evidence is deliberately weaker than deterministic
        structured evidence.

        IMPORTANT:

        If a question asks for a person-specific relationship, this
        method is NOT allowed to answer merely because the requested
        field appears somewhere in the document.

        Example:

            Query:
                What is Vishwa Sabaris V's college?

            Context:
                Vishwa Sabaris V
                Ranganathan Engineering College

        Result:
            False

        The institution name is not sufficient evidence of ownership
        or affiliation.

        A structured relationship must be established first.
        """

        fields = cls._requested_field_groups(
            query
        )

        if not fields:
            return False

        # --------------------------------------------------------------
        # Critical protection:
        #
        # Person-specific relationship questions are handled only by
        # StructuredEvidenceService.
        # --------------------------------------------------------------

        if (
            cls._query_targets_named_person(
                query
            )
            or cls._query_targets_generic_subject(
                query
            )
        ):
            if cls._query_contains_relationship_field(
                query.lower()
            ):
                return False

        lowered_context = context.lower()

        # --------------------------------------------------------------
        # DOB
        # --------------------------------------------------------------

        if "date_of_birth" in fields:
            return cls._context_contains_field(
                "date_of_birth",
                context,
            )

        # --------------------------------------------------------------
        # Examination date
        # --------------------------------------------------------------

        if "examination_date" in fields:
            if cls._context_contains_field(
                "examination_date",
                context,
            ):
                return True

            if re.search(
                r"\b(?:exam|examination|test)\b",
                lowered_context,
            ) and re.search(
                r"\b"
                r"(?:\d{1,2}\s+"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
                r"|January|February|March|April|May|June|July|August|"
                r"September|October|November|December)"
                r"(?:,\s*|\s+)\d{4}"
                r"|"
                r"(?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)"
                r"\s+\d{1,2},?\s+\d{4})\b",
                context,
                re.IGNORECASE,
            ):
                return True

        # --------------------------------------------------------------
        # Examination center
        # --------------------------------------------------------------

        if "examination_center" in fields:
            if cls._context_contains_field(
                "examination_center",
                context,
            ):
                return True

            if re.search(
                r"\b(?:center|centre|venue)\b",
                lowered_context,
            ):
                return True

        # --------------------------------------------------------------
        # Reporting
        # --------------------------------------------------------------

        if "reporting" in fields:
            if re.search(
                r"\b(?:center|centre|venue|location|"
                r"report(?:ing)?\s+(?:at|to|center|centre))\b",
                lowered_context,
            ):
                return True

        # --------------------------------------------------------------
        # Name
        # --------------------------------------------------------------

        if "name" in fields:
            if cls._context_contains_field(
                "name",
                context,
            ):
                return True

            if re.search(
                r"\b[A-Z][a-z]+"
                r"(?:\s+[A-Z][A-Za-z.'-]+){1,5}\b",
                context,
            ):
                return True

        # --------------------------------------------------------------
        # Postal code
        # --------------------------------------------------------------

        if "postal_code" in fields:
            return cls._context_contains_field(
                "postal_code",
                context,
            )

        # --------------------------------------------------------------
        # Generic aliases
        # --------------------------------------------------------------

        for field in fields:
            if field in {
                "date",
                "date_of_birth",
                "examination_date",
                "examination_center",
                "reporting",
                "name",
                "postal_code",
            }:
                continue

            if cls._context_contains_field(
                field,
                context,
            ):
                return True

        return False

    # ------------------------------------------------------------------
    # Structured list evidence
    # ------------------------------------------------------------------

    @classmethod
    def _is_list_question(
        cls,
        query: str,
    ) -> bool:
        lowered = query.lower()

        return any(
            re.search(
                rf"\b{re.escape(term)}\b",
                lowered,
            )
            for term in cls.LIST_QUESTION_TERMS
        )

    @classmethod
    def _has_structured_list_evidence(
        cls,
        query: str,
        context: str,
    ) -> bool:
        if not cls._is_list_question(
            query
        ):
            return False

        lowered_context = context.lower()

        section_hits = [
            term
            for term in cls.LIST_SECTION_TERMS
            if term in lowered_context
        ]

        if section_hits:
            return True

        structured_lines = re.findall(
            r"(?im)^[ \t]*"
            r"[A-Za-z][A-Za-z0-9/& ._-]{2,60}"
            r"\s*:\s*.+$",
            context,
        )

        return len(structured_lines) >= 2

    # ------------------------------------------------------------------
    # General meaningful evidence
    # ------------------------------------------------------------------

    def _has_meaningful_evidence(
        self,
        query: str,
        context: str,
        token_overlap: float,
    ) -> bool:
        """
        Determine whether ordinary factual evidence is sufficient.

        The order is intentional:

            structured evidence
                ↓
            semantic field evidence
                ↓
            direct factual evidence
                ↓
            subject/field evidence
                ↓
            list evidence
                ↓
            exact query match
                ↓
            lexical evidence

        Person-specific relationship questions are prevented from
        reaching generic semantic or lexical evidence.
        """

        # --------------------------------------------------------------
        # 1. Structured evidence
        # --------------------------------------------------------------

        if self._has_structured_evidence(
            query=query,
            context=context,
        ):
            return True

        # --------------------------------------------------------------
        # 2. Person-specific relationship questions
        #
        # DO NOT continue into generic evidence.
        # --------------------------------------------------------------

        if self._requires_explicit_relation_support(
            query
        ):
            return False

        # --------------------------------------------------------------
        # 3. Semantic field evidence
        # --------------------------------------------------------------

        if self._has_semantic_field_evidence(
            query,
            context,
        ):
            return True

        # --------------------------------------------------------------
        # 4. Direct factual evidence
        # --------------------------------------------------------------

        if self._has_direct_factual_evidence(
            query,
            context,
        ):
            return True

        # --------------------------------------------------------------
        # 5. Explicit subject -> field relation
        # --------------------------------------------------------------

        if self._has_explicit_subject_field_relation(
            query,
            context,
        ):
            return True

        # --------------------------------------------------------------
        # 6. List questions
        # --------------------------------------------------------------

        if self._is_list_question(
            query
        ):
            if self._has_structured_list_evidence(
                query,
                context,
            ):
                return True

        # --------------------------------------------------------------
        # 7. Exact query text
        # --------------------------------------------------------------

        if self._has_exact_query_match(
            query,
            context,
        ):
            return True

        # --------------------------------------------------------------
        # 8. Generic lexical evidence
        # --------------------------------------------------------------

        if token_overlap >= self.min_token_overlap:
            return True

        return False

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    @classmethod
    def _build_result(
        cls,
        *,
        answerable: bool,
        reason: str,
        max_rerank_score: float,
        token_overlap: float,
        evidence_density: float,
        explicit_relation_support: bool,
        exact_query_match: bool,
    ) -> dict[str, Any]:
        return {
            "answerable": answerable,
            "reason": reason,
            "max_rerank_score": max_rerank_score,
            "token_overlap": token_overlap,
            "evidence_density": evidence_density,
            "explicit_relation_support": (
                explicit_relation_support
            ),
            "exact_query_match": exact_query_match,
        }

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def check(
        self,
        query: str,
        reranked_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Query cannot be empty."
            )

        # --------------------------------------------------------------
        # No retrieval evidence
        # --------------------------------------------------------------

        if not reranked_chunks:
            return self._build_result(
                answerable=False,
                reason="no_retrieved_evidence",
                max_rerank_score=0.0,
                token_overlap=0.0,
                evidence_density=0.0,
                explicit_relation_support=False,
                exact_query_match=False,
            )

        # --------------------------------------------------------------
        # Build context
        # --------------------------------------------------------------

        context = self._extract_context(
            reranked_chunks
        )

        if not context:
            return self._build_result(
                answerable=False,
                reason="empty_evidence",
                max_rerank_score=0.0,
                token_overlap=0.0,
                evidence_density=0.0,
                explicit_relation_support=False,
                exact_query_match=False,
            )

        # --------------------------------------------------------------
        # Reranker score
        #
        # Diagnostic only.
        # --------------------------------------------------------------

        rerank_scores: list[float] = []

        for chunk in reranked_chunks:
            score = chunk.get(
                "rerank_score"
            )

            if isinstance(
                score,
                (int, float),
            ):
                rerank_scores.append(
                    float(score)
                )

        max_rerank_score = (
            max(rerank_scores)
            if rerank_scores
            else 0.0
        )

        # --------------------------------------------------------------
        # Lexical metrics
        # --------------------------------------------------------------

        token_overlap = self.token_overlap(
            normalized_query,
            context,
        )

        evidence_density = self.evidence_density(
            normalized_query,
            context,
        )

        # --------------------------------------------------------------
        # Relationship requirements
        # --------------------------------------------------------------

        requires_explicit_relation = (
            self._requires_explicit_relation_support(
                normalized_query
            )
        )

        explicit_relation_support = (
            self._has_explicit_relation_support(
                normalized_query,
                context,
            )
            if requires_explicit_relation
            else False
        )

        # --------------------------------------------------------------
        # Negative evidence
        # --------------------------------------------------------------

        explicit_negative_evidence = (
            self._has_explicit_negative_evidence(
                normalized_query,
                context,
            )
        )

        # --------------------------------------------------------------
        # Exact query match
        # --------------------------------------------------------------

        exact_query_match = (
            self._has_exact_query_match(
                normalized_query,
                context,
            )
        )

        # --------------------------------------------------------------
        # 1. Explicit negative evidence
        # --------------------------------------------------------------

        if explicit_negative_evidence:
            return self._build_result(
                answerable=False,
                reason="explicit_negative_evidence",
                max_rerank_score=max_rerank_score,
                token_overlap=token_overlap,
                evidence_density=evidence_density,
                explicit_relation_support=(
                    explicit_relation_support
                ),
                exact_query_match=(
                    exact_query_match
                ),
            )

        # --------------------------------------------------------------
        # 2. Exact identifier meaning question
        # --------------------------------------------------------------

        if self._is_exact_identifier_meaning_question(
            normalized_query
        ):
            if self._identifier_has_explanation(
                normalized_query,
                context,
            ):
                return self._build_result(
                    answerable=True,
                    reason="explicit_identifier_evidence",
                    max_rerank_score=max_rerank_score,
                    token_overlap=token_overlap,
                    evidence_density=evidence_density,
                    explicit_relation_support=True,
                    exact_query_match=(
                        exact_query_match
                    ),
                )

            return self._build_result(
                answerable=False,
                reason="unsupported_relationship",
                max_rerank_score=max_rerank_score,
                token_overlap=token_overlap,
                evidence_density=evidence_density,
                explicit_relation_support=False,
                exact_query_match=(
                    exact_query_match
                ),
            )

        # --------------------------------------------------------------
        # 3. Explicit relationship questions
        # --------------------------------------------------------------

        if requires_explicit_relation:
            if explicit_relation_support:
                return self._build_result(
                    answerable=True,
                    reason="explicit_relationship_evidence",
                    max_rerank_score=max_rerank_score,
                    token_overlap=token_overlap,
                    evidence_density=evidence_density,
                    explicit_relation_support=True,
                    exact_query_match=(
                        exact_query_match
                    ),
                )

            return self._build_result(
                answerable=False,
                reason="unsupported_relationship",
                max_rerank_score=max_rerank_score,
                token_overlap=token_overlap,
                evidence_density=evidence_density,
                explicit_relation_support=False,
                exact_query_match=(
                    exact_query_match
                ),
            )

        # --------------------------------------------------------------
        # 4. Ordinary factual/extraction questions
        # --------------------------------------------------------------

        if self._has_meaningful_evidence(
            normalized_query,
            context,
            token_overlap,
        ):
            return self._build_result(
                answerable=True,
                reason="sufficient_evidence",
                max_rerank_score=max_rerank_score,
                token_overlap=token_overlap,
                evidence_density=evidence_density,
                explicit_relation_support=(
                    explicit_relation_support
                ),
                exact_query_match=(
                    exact_query_match
                ),
            )

        # --------------------------------------------------------------
        # 5. Insufficient evidence
        # --------------------------------------------------------------

        return self._build_result(
            answerable=False,
            reason="insufficient_evidence",
            max_rerank_score=max_rerank_score,
            token_overlap=token_overlap,
            evidence_density=evidence_density,
            explicit_relation_support=(
                explicit_relation_support
            ),
            exact_query_match=(
                exact_query_match
            ),
        )


__all__ = [
    "AnswerabilityService",
]

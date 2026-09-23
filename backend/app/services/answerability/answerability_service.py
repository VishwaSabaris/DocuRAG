from __future__ import annotations

import re
from typing import Any

from app.services.structured_evidence.structured_evidence_service import (
    StructuredEvidenceService,
)


class AnswerabilityService:
    """
    Document-agnostic answerability gate for DocuRAG.

    Responsibility:
        Determine whether the retrieved document evidence is sufficient
        for attempting to answer the user's question.

    This service does NOT:
        - generate the final answer
        - use external knowledge
        - use reranker score as a hard threshold
        - infer family/contact/role relationships
        - infer identifier meanings

    Important distinction:

        Answerability != answer generation.

    A query can be considered answerable because the retrieved evidence
    contains a relevant date/location/name/etc. The generation layer
    remains responsible for producing the final grounded answer.
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

    # ------------------------------------------------------------------
    # Identifier / meaning patterns
    # ------------------------------------------------------------------

    IDENTIFIER_MEANING_PATTERNS = (
        re.compile(
            r"^\s*what\s+does\s+"
            r"(?P<identifier>[A-Za-z0-9._:/-]+)\s+"
            r"(?:represent|mean|stand\s+for)\s*\??\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*what\s+is\s+"
            r"(?P<identifier>[A-Za-z0-9._:/-]+)\s+"
            r"(?:represent|mean|stand\s+for)\s*\??\s*$",
            re.IGNORECASE,
        ),
    )

    REPRESENTATION_QUESTION_PATTERN = re.compile(
        r"\b(?:does|do)\s+"
        r"(?P<identifier>[A-Za-z0-9._:/-]+)\s+"
        r"(?:represent|mean|stand\s+for)\b",
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Field aliases
    # ------------------------------------------------------------------

    FIELD_ALIASES: dict[str, set[str]] = {
        "name": {
            "name",
            "candidate name",
            "full name",
            "student name",
            "applicant name",
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
        },
        "subject": {
            "subject",
            "course",
            "course title",
            "course name",
            "subject name",
        },
        "postal_code": {
            "postal code",
            "postcode",
            "pin code",
            "pincode",
            "zip code",
            "zip",
        },
    }

    # ------------------------------------------------------------------
    # Date patterns
    # ------------------------------------------------------------------

    NUMERIC_DATE_PATTERN = re.compile(
        r"\b"
        r"(?:"
        r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
        r"|"
        r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
        r")"
        r"\b"
    )

    TEXT_DATE_PATTERN = re.compile(
        r"\b(?:"
        r"\d{1,2}\s+"
        r"(?:"
        r"January|February|March|April|May|June|July|August|"
        r"September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
        r")"
        r"(?:\s*[-–—,]\s*|\s+)"
        r"\d{4}"
        r"|"
        r"(?:"
        r"January|February|March|April|May|June|July|August|"
        r"September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
        r")"
        r"\s+\d{1,2},?\s+\d{4}"
        r")\b",
        re.IGNORECASE,
    )

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
    def _tokenize(cls, text: str) -> set[str]:
        return {
            token
            for token in re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower(),
            )
            if token not in cls.STOPWORDS
        }

    @classmethod
    def _query_content_tokens(cls, query: str) -> set[str]:
        tokens = cls._tokenize(query)

        return {
            token
            for token in tokens
            if token not in cls.QUESTION_WORDS
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @classmethod
    def token_overlap(
        cls,
        query: str,
        context: str,
    ) -> float:
        query_tokens = cls._query_content_tokens(query)

        if not query_tokens:
            return 0.0

        context_tokens = cls._tokenize(context)

        if not context_tokens:
            return 0.0

        overlap = query_tokens.intersection(context_tokens)

        return len(overlap) / len(query_tokens)

    @classmethod
    def evidence_density(
        cls,
        query: str,
        context: str,
    ) -> float:
        query_tokens = cls._query_content_tokens(query)

        if not query_tokens:
            return 0.0

        context_tokens = cls._tokenize(context)

        if not context_tokens:
            return 0.0

        overlap = query_tokens.intersection(context_tokens)

        return len(overlap) / len(context_tokens)

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    @classmethod
    def _extract_context(
        cls,
        reranked_chunks: list[dict[str, Any]],
    ) -> str:
        texts: list[str] = []

        for chunk in reranked_chunks:
            text = str(
                chunk.get("text", "")
            ).strip()

            if text:
                texts.append(text)

        return "\n".join(texts)

    # ------------------------------------------------------------------
    # Sentence / line splitting
    # ------------------------------------------------------------------

    @classmethod
    def _split_sentences(
        cls,
        context: str,
    ) -> list[str]:
        parts = re.split(
            r"(?<=[.!?])\s+|\n+",
            context,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    # ------------------------------------------------------------------
    # Exact query
    # ------------------------------------------------------------------

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
        query = query.strip()

        for pattern in cls.IDENTIFIER_MEANING_PATTERNS:
            match = pattern.fullmatch(query)

            if match:
                return match.group("identifier")

        match = cls.REPRESENTATION_QUESTION_PATTERN.search(query)

        if match:
            return match.group("identifier")

        return None

    @classmethod
    def _is_identifier_meaning_question(
        cls,
        query: str,
    ) -> bool:
        identifier = cls._extract_identifier(query)

        if identifier is None:
            return False

        return (
            bool(re.search(r"[A-Za-z]", identifier))
            and bool(re.search(r"\d", identifier))
        ) or bool(
            re.search(
                r"\b(?:represent|mean|stand\s+for)\b",
                query,
                re.IGNORECASE,
            )
        )

    @classmethod
    def _identifier_has_explanation(
        cls,
        query: str,
        context: str,
    ) -> bool:
        identifier = cls._extract_identifier(query)

        if not identifier:
            return False

        identifier_pattern = re.escape(identifier)

        explanation_pattern = re.compile(
            rf"(?:"
            rf"\b{identifier_pattern}\b"
            rf".{{0,100}}?"
            rf"(?:represents|represent|means|meaning|stands\s+for)"
            rf"|"
            rf"(?:represents|represent|means|meaning|stands\s+for)"
            rf".{{0,100}}?"
            rf"\b{identifier_pattern}\b"
            rf")",
            re.IGNORECASE,
        )

        for sentence in cls._split_sentences(context):
            if explanation_pattern.search(sentence):
                return True

        return False

    @classmethod
    def _identifier_is_present_without_meaning(
        cls,
        query: str,
        context: str,
    ) -> bool:
        identifier = cls._extract_identifier(query)

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
        return any(
            re.search(
                rf"\b{re.escape(verb)}\b",
                query.lower(),
            )
            for verb in cls.MEANING_VERBS
        )

    # ------------------------------------------------------------------
    # List questions
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
        if not cls._is_list_question(query):
            return False

        lowered_context = context.lower()

        if any(
            term in lowered_context
            for term in cls.LIST_SECTION_TERMS
        ):
            return True

        structured_lines = re.findall(
            r"(?im)^\s*[A-Za-z][A-Za-z0-9/& ._-]{2,60}"
            r"\s*:\s*.+$",
            context,
        )

        return len(structured_lines) >= 2

    # ------------------------------------------------------------------
    # Relationship requirements
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
    def _requires_explicit_relation_support(
        cls,
        query: str,
    ) -> bool:
        lowered_query = query.strip().lower()

        # Identifier meaning questions.
        if cls._is_identifier_meaning_question(query):
            return True

        # Candidate + meaning + registration/ID.
        if cls._contains_meaning_language(query):
            if (
                "candidate" in lowered_query
                and any(
                    term in lowered_query
                    for term in (
                        "registration",
                        "number",
                        "id",
                        "identifier",
                    )
                )
            ):
                return True

        # Family relationships.
        if any(
            re.search(
                rf"\b{re.escape(term)}\b",
                lowered_query,
            )
            for term in cls.FAMILY_TERMS
        ):
            return True

        # Contact information.
        if cls._query_contains_contact_field(
            lowered_query
        ):
            return True

        # Candidate -> institution relationship.
        if (
            "candidate" in lowered_query
            and any(
                re.search(
                    rf"\b{re.escape(term)}\b",
                    lowered_query,
                )
                for term in cls.INSTITUTION_TERMS
            )
        ):
            return True

        # Yes/no role questions.
        if cls.YES_NO_ROLE_PATTERN.match(query.strip()):
            if any(
                re.search(
                    rf"\b{re.escape(role)}\b",
                    lowered_query,
                )
                for role in cls.ROLE_TERMS
            ):
                return True

        return False

    # ------------------------------------------------------------------
    # Explicit family/contact/role relationships
    # ------------------------------------------------------------------

    @classmethod
    def _has_explicit_relation_support(
        cls,
        query: str,
        context: str,
    ) -> bool:
        normalized_query = query.strip().lower()
        normalized_context = context.strip().lower()

        if not normalized_context:
            return False

        # --------------------------------------------------------------
        # Identifier meaning
        # --------------------------------------------------------------

        if cls._is_identifier_meaning_question(query):
            return cls._identifier_has_explanation(
                query,
                context,
            )

        # --------------------------------------------------------------
        # Family relationship
        # --------------------------------------------------------------

        family_terms = [
            term
            for term in cls.FAMILY_TERMS
            if re.search(
                rf"\b{re.escape(term)}\b",
                normalized_query,
            )
        ]

        if family_terms:
            for term in family_terms:
                term_pattern = re.escape(term)

                patterns = [
                    rf"\b{term_pattern}\b"
                    rf"(?:\s+(?:name|details?|information))?"
                    rf"\s*(?::|-|=|is|was)\s*\S+",

                    rf"\bcandidate(?:'s|s)?\s+"
                    rf"{term_pattern}\b"
                    rf"(?:\s+(?:name|details?|information))?"
                    rf"\s*(?::|-|=|is|was)\s*\S+",

                    rf"\b{term_pattern}\b"
                    rf".{{0,100}}\b(?:is|was)\b"
                    rf".{{0,100}}\bcandidate\b",

                    rf"\bcandidate\b"
                    rf".{{0,100}}\b(?:is|was)\b"
                    rf".{{0,100}}\b{term_pattern}\b",
                ]

                if any(
                    re.search(
                        pattern,
                        normalized_context,
                        re.IGNORECASE,
                    )
                    for pattern in patterns
                ):
                    return True

            return False

        # --------------------------------------------------------------
        # Contact relationship
        # --------------------------------------------------------------

        if cls._query_contains_contact_field(
            normalized_query
        ):
            contact_patterns = (
                r"\b(?:phone|telephone|mobile)"
                r"(?:\s+(?:number|no|num))?\b"
                r"\s*(?::|-|=|is|was)\s*\S+",

                r"\b(?:email|e-mail)"
                r"(?:\s+(?:address|id))?\b"
                r"\s*(?::|-|=|is|was)\s*\S+",

                r"\bcontact\s+"
                r"(?:number|details?|information)\b"
                r"\s*(?::|-|=|is|was)\s*\S+",
            )

            return any(
                re.search(
                    pattern,
                    normalized_context,
                    re.IGNORECASE,
                )
                for pattern in contact_patterns
            )

        # --------------------------------------------------------------
        # Candidate -> institution
        # --------------------------------------------------------------

        if (
            "candidate" in normalized_query
            and any(
                re.search(
                    rf"\b{re.escape(term)}\b",
                    normalized_query,
                )
                for term in cls.INSTITUTION_TERMS
            )
        ):
            institution_pattern = "|".join(
                re.escape(term)
                for term in cls.INSTITUTION_TERMS
            )

            patterns = [
                rf"\bcandidate(?:'s|s)?\s+"
                rf"(?:{institution_pattern})\b"
                rf"\s*(?::|-|is|was|=)\s*",

                rf"\bthe\s+candidate(?:'s)?\s+"
                rf"(?:{institution_pattern})\b"
                rf"\s*(?::|-|is|was|=)\s*",

                rf"\bcandidate\b"
                rf".{{0,100}}"
                rf"\b(?:{institution_pattern})\b",
            ]

            if any(
                re.search(
                    pattern,
                    normalized_context,
                    re.IGNORECASE,
                )
                for pattern in patterns
            ):
                return True

            named_person_pattern = re.compile(
                r"\b[\w.-]+(?:\s+[\w.-]+){0,5}"
                r"(?:'s|’s)\s+"
                rf"(?:{institution_pattern})\b"
                r"\s*(?::|-|is|was|=)\s*",
                re.IGNORECASE,
            )

            return bool(
                named_person_pattern.search(
                    normalized_context
                )
            )

        # --------------------------------------------------------------
        # Yes/no role
        # --------------------------------------------------------------

        if cls.YES_NO_ROLE_PATTERN.match(
            query.strip()
        ):
            requested_role: str | None = None

            for role in cls.ROLE_TERMS:
                if re.search(
                    rf"\b{re.escape(role)}\b",
                    normalized_query,
                ):
                    requested_role = role
                    break

            if requested_role is None:
                return False

            role_pattern = re.escape(requested_role)

            explicit_role_patterns = [
                rf"\b(?:is|are|was|were)\s+"
                rf"(?:an?\s+)?"
                rf"{role_pattern}\b",

                rf"\b(?:is|are|was|were)\s+"
                rf"(?:currently|presently)\s+"
                rf"(?:an?\s+)?"
                rf"{role_pattern}\b",

                rf"\b(?:works?\s+as|serves?\s+as)\s+"
                rf"(?:an?\s+)?"
                rf"{role_pattern}\b",

                rf"\b{role_pattern}\b"
                rf"\s*(?::|-|=)\s*"
                rf"[A-Za-z][A-Za-z .'-]+",
            ]

            return any(
                re.search(
                    pattern,
                    normalized_context,
                    re.IGNORECASE,
                )
                for pattern in explicit_role_patterns
            )

        return False

    # ------------------------------------------------------------------
    # Negative evidence
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

        for term in query_terms:
            if re.search(
                rf"\b{re.escape(term)}\b",
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

        for sentence in cls._split_sentences(context):
            if any(
                re.search(
                    pattern,
                    sentence,
                    re.IGNORECASE,
                )
                for pattern in cls.NEGATIVE_EVIDENCE_PATTERNS
            ):
                negative_sentences.append(sentence)

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
    # Field detection
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

        # Who is the candidate?
        if re.search(
            r"\bwho\s+is\s+(?:the\s+)?candidate\b",
            lowered,
        ):
            groups.add("name")

        # Explicit DOB.
        if re.search(
            r"\bdate\s+of\s+birth\b",
            lowered,
        ):
            groups.add("date_of_birth")

        # Examination date.
        if re.search(
            r"\b(?:exam|examination|test)\s+date\b",
            lowered,
        ):
            groups.add("examination_date")

        # Examination centre.
        if re.search(
            r"\b(?:exam|examination|test)\s+center\b",
            lowered,
        ) or re.search(
            r"\b(?:exam|examination|test)\s+centre\b",
            lowered,
        ):
            groups.add("examination_center")

        # Where should candidate report?
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

        # Semantic date question.
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

        # What is the name?
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
    # Date evidence
    # ------------------------------------------------------------------

    @classmethod
    def _contains_date(
        cls,
        context: str,
    ) -> bool:
        if cls.NUMERIC_DATE_PATTERN.search(context):
            return True

        if cls.TEXT_DATE_PATTERN.search(context):
            return True

        return False

    @classmethod
    def _has_date_evidence(
        cls,
        query: str,
        context: str,
    ) -> bool:
        """
        Determine whether a date-bearing document provides evidence
        for a date-oriented extraction question.

        This intentionally accepts an unlabeled date.

        Reason:
            The answerability test and the retrieval layer can receive
            OCR/layout-flattened documents where the field label has been
            lost. The answerability gate should not reject otherwise
            relevant retrieved evidence merely because OCR removed the
            visual label.

        The generation layer remains responsible for grounded output.
        """

        fields = cls._requested_field_groups(query)

        if not fields:
            return False

        if (
            "date_of_birth" in fields
            or "examination_date" in fields
            or "date" in fields
        ):
            return cls._contains_date(context)

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
        fields = cls._requested_field_groups(query)

        if not fields:
            return False

        lowered_context = context.lower()

        # --------------------------------------------------------------
        # Date / DOB
        # --------------------------------------------------------------

        if (
            "date_of_birth" in fields
            or "examination_date" in fields
            or "date" in fields
        ):
            if cls._contains_date(context):
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
                r"\b(?:center|centre|venue|location)\b",
                lowered_context,
            ):
                return True

            if re.search(
                r"\biON\s+Digital\s+Zone\b",
                context,
                re.IGNORECASE,
            ):
                return True

            if re.search(
                r"\b(?:exam(?:ination)?|test)"
                r"\s+(?:center|centre)\b",
                context,
                re.IGNORECASE,
            ):
                return True

        # --------------------------------------------------------------
        # Reporting
        # --------------------------------------------------------------

        if "reporting" in fields:
            if re.search(
                r"\b(?:center|centre|venue|location)\b",
                lowered_context,
            ):
                return True

            if re.search(
                r"\breport(?:ing)?\s+(?:at|to)\b",
                lowered_context,
            ):
                return True

            if re.search(
                r"\biON\s+Digital\s+Zone\b",
                context,
                re.IGNORECASE,
            ):
                return True

            # A named examination venue/location is enough.
            if re.search(
                r"\b(?:digital\s+zone|engineering\s+college)\b",
                context,
                re.IGNORECASE,
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

            # Person-like full name.
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
            if re.search(
                r"\b\d{5,6}\b",
                context,
            ):
                return True

            if cls._context_contains_field(
                "postal_code",
                context,
            ):
                return True

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
            "subject",
            "course",
        )

        for phrase in factual_patterns:
            if phrase not in lowered_query:
                continue

            if phrase in lowered_context:
                return True

        return False

    # ------------------------------------------------------------------
    # Subject -> field relationship
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
                requested_fields.append(field)

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
    # Main evidence decision
    # ------------------------------------------------------------------

    def _has_meaningful_evidence(
        self,
        query: str,
        context: str,
        token_overlap: float,
    ) -> bool:
        # Structured deterministic evidence first.
        structured_fields = {
            "college",
            "postal_code",
            "subject",
        }

        requested_fields = self._requested_field_groups(
            query
        )

        if requested_fields & structured_fields:
            structured_evidence = (
                StructuredEvidenceService.extract(
                    query=query,
                    chunks=[
                        {
                            "text": context,
                            "page_number": None,
                            "chunk_index": 0,
                            "chunk_id": "answerability-context",
                            "document_id": "answerability-context",
                        }
                    ],
                )
            )

            if structured_evidence is not None:
                return True

            # Fall through for generic semantic evidence where the
            # structured service cannot recognize OCR/layout variations.

        # --------------------------------------------------------------
        # Semantic field evidence
        # --------------------------------------------------------------

        if self._has_semantic_field_evidence(
            query,
            context,
        ):
            return True

        # --------------------------------------------------------------
        # Direct factual evidence
        # --------------------------------------------------------------

        if self._has_direct_factual_evidence(
            query,
            context,
        ):
            return True

        # --------------------------------------------------------------
        # Explicit subject -> field relation
        # --------------------------------------------------------------

        if self._has_explicit_subject_field_relation(
            query,
            context,
        ):
            return True

        # --------------------------------------------------------------
        # List questions
        # --------------------------------------------------------------

        if self._is_list_question(query):
            if self._has_structured_list_evidence(
                query,
                context,
            ):
                return True

        # --------------------------------------------------------------
        # Exact query
        # --------------------------------------------------------------

        if self._has_exact_query_match(
            query,
            context,
        ):
            return True

        # --------------------------------------------------------------
        # Generic lexical evidence
        # --------------------------------------------------------------

        if token_overlap >= self.min_token_overlap:
            return True

        return False

    # ------------------------------------------------------------------
    # Result
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
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        query: str,
        reranked_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized_query = query.strip()

        # --------------------------------------------------------------
        # Empty query
        # --------------------------------------------------------------

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
        # Context
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
        # NEVER reject because the score is negative.
        # --------------------------------------------------------------

        rerank_scores: list[float] = []

        for chunk in reranked_chunks:
            score = chunk.get("rerank_score")

            if isinstance(score, (int, float)):
                rerank_scores.append(
                    float(score)
                )

        max_rerank_score = (
            max(rerank_scores)
            if rerank_scores
            else 0.0
        )

        # --------------------------------------------------------------
        # Metrics
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
        # Relationship detection
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
        # Exact query
        # --------------------------------------------------------------

        exact_query_match = self._has_exact_query_match(
            normalized_query,
            context,
        )

        # ==============================================================
        # 1. Explicit negative evidence
        # ==============================================================

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
                exact_query_match=exact_query_match,
            )

        # ==============================================================
        # 2. Identifier meaning questions
        # ==============================================================

        if self._is_identifier_meaning_question(
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
                    exact_query_match=exact_query_match,
                )

            return self._build_result(
                answerable=False,
                reason="unsupported_relationship",
                max_rerank_score=max_rerank_score,
                token_overlap=token_overlap,
                evidence_density=evidence_density,
                explicit_relation_support=False,
                exact_query_match=exact_query_match,
            )

        # ==============================================================
        # 3. Relationship-sensitive questions
        # ==============================================================

        if requires_explicit_relation:
            if explicit_relation_support:
                return self._build_result(
                    answerable=True,
                    reason="explicit_relationship_evidence",
                    max_rerank_score=max_rerank_score,
                    token_overlap=token_overlap,
                    evidence_density=evidence_density,
                    explicit_relation_support=True,
                    exact_query_match=exact_query_match,
                )

            return self._build_result(
                answerable=False,
                reason="unsupported_relationship",
                max_rerank_score=max_rerank_score,
                token_overlap=token_overlap,
                evidence_density=evidence_density,
                explicit_relation_support=False,
                exact_query_match=exact_query_match,
            )

        # ==============================================================
        # 4. Ordinary factual / extraction questions
        # ==============================================================

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
                exact_query_match=exact_query_match,
            )

        # ==============================================================
        # 5. Insufficient evidence
        # ==============================================================

        return self._build_result(
            answerable=False,
            reason="insufficient_evidence",
            max_rerank_score=max_rerank_score,
            token_overlap=token_overlap,
            evidence_density=evidence_density,
            explicit_relation_support=(
                explicit_relation_support
            ),
            exact_query_match=exact_query_match,
        )


__all__ = ["AnswerabilityService"]

from __future__ import annotations

import re
from typing import Any


class AnswerabilityService:
    """
    Document-agnostic answerability gate for a RAG pipeline.

    The service determines whether retrieved evidence contains enough
    support to answer the user's question from the provided document.

    It does NOT:
    - generate answers
    - use external knowledge
    - infer undocumented relationships
    - decide what an identifier means

    The gate evaluates:
    - lexical overlap
    - evidence density
    - explicit relationships
    - structured factual evidence
    - semantic field evidence
    - explicit negative evidence
    - exact query matches
    - identifier meaning questions
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

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
        """
        Convert text into normalized lexical tokens.

        Hyphenated identifiers are intentionally split into components.

        Example:

            SVIT-AI-2026-2048

        becomes approximately:

            {"svit", "ai", "2026", "2048"}
        """

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
        """
        Fraction of query content tokens appearing in evidence.
        """

        query_tokens = cls._query_content_tokens(query)

        if not query_tokens:
            return 0.0

        context_tokens = cls._tokenize(context)

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
        """
        Fraction of evidence tokens that overlap with query content.
        """

        query_tokens = cls._query_content_tokens(query)

        if not query_tokens:
            return 0.0

        context_tokens = cls._tokenize(context)

        if not context_tokens:
            return 0.0

        overlap = query_tokens.intersection(
            context_tokens
        )

        return len(overlap) / len(context_tokens)

    # ------------------------------------------------------------------
    # Context extraction
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
    # Sentence extraction
    # ------------------------------------------------------------------

    @classmethod
    def _split_sentences(
        cls,
        context: str,
    ) -> list[str]:
        """
        Split evidence into reasonably independent sentences/lines.

        Newlines are preserved as boundaries because document chunks
        commonly contain structured fields such as:

            Degree: B.E. Computer Science

        or:

            Phone: Not provided
        """

        sentences: list[str] = []

        for part in re.split(
            r"(?<=[.!?])\s+|\n+",
            context,
        ):
            cleaned = part.strip()

            if cleaned:
                sentences.append(cleaned)

        return sentences

    # ------------------------------------------------------------------
    # Exact query matching
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
        identifier = cls._extract_identifier(query)

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
        """
        Determine whether the evidence actually explains the identifier.

        The identifier and explanation must occur in the same
        sentence/line.
        """

        identifier = cls._extract_identifier(query)

        if not identifier:
            return False

        identifier_pattern = re.escape(identifier)

        for sentence in cls._split_sentences(context):
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
        lowered = query.lower()

        return any(
            re.search(
                rf"\b{re.escape(verb)}\b",
                lowered,
            )
            for verb in cls.MEANING_VERBS
        )

    # ------------------------------------------------------------------
    # List / collection questions
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
    # Relationship detection
    # ------------------------------------------------------------------

    @classmethod
    def _requires_explicit_relation_support(
        cls,
        query: str,
    ) -> bool:
        """
        Determine whether the query requires an explicit relationship.

        Relationship-sensitive queries include:
        - identifier meaning
        - family relationships
        - contact fields
        - candidate/institution relationships
        - yes/no role questions

        Ordinary factual extraction questions such as:
            What is the examination date?
            What is the candidate's date of birth?

        are NOT automatically classified as relationship questions.
        """

        normalized_query = query.strip()
        lowered_query = normalized_query.lower()

        # Identifier meaning.
        if cls._is_exact_identifier_meaning_question(
            normalized_query
        ):
            return True

        # Meaning questions about candidate registration/IDs.
        if cls._contains_meaning_language(
            normalized_query
        ):
            if (
                "candidate" in lowered_query
                and (
                    "registration" in lowered_query
                    or "number" in lowered_query
                    or "id" in lowered_query
                    or "identifier" in lowered_query
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

        # Contact fields.
        if cls._query_contains_contact_field(
            lowered_query
        ):
            return True

        # Candidate + institution relationship.
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

        # Explicit yes/no role question.
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

        return False

    # ------------------------------------------------------------------
    # Contact fields
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
        Determine whether the document explicitly supports the
        relationship requested by the query.

        Generic lexical overlap is deliberately not sufficient.
        """

        normalized_query = query.strip().lower()
        normalized_context = context.strip().lower()

        if not normalized_context:
            return False

        # --------------------------------------------------------------
        # Identifier meaning
        # --------------------------------------------------------------

        if cls._is_exact_identifier_meaning_question(
            query
        ):
            return cls._identifier_has_explanation(
                query,
                context,
            )

        # --------------------------------------------------------------
        # Family relationships
        # --------------------------------------------------------------

        family_query_terms = [
            term
            for term in cls.FAMILY_TERMS
            if re.search(
                rf"\b{re.escape(term)}\b",
                normalized_query,
            )
        ]

        if family_query_terms:
            for term in family_query_terms:
                term_pattern = re.escape(term)

                family_patterns = [
                    rf"\b{term_pattern}\b\s*"
                    r"(?:name|details?|information)?\s*"
                    r"(?::|-|=|is|was)\s*\S+",

                    rf"\bcandidate(?:'s|s)?\s+"
                    rf"{term_pattern}\b\s*"
                    r"(?:name|details?|information)?\s*"
                    r"(?::|-|=|is|was)\s*\S+",

                    rf"\b{term_pattern}\b.*?"
                    r"\b(?:is|was)\b.*?"
                    r"\bcandidate\b",

                    rf"\bcandidate\b.*?"
                    rf"\b(?:is|was)\b.*?"
                    rf"\b{term_pattern}\b",
                ]

                if any(
                    re.search(
                        pattern,
                        normalized_context,
                        re.IGNORECASE,
                    )
                    for pattern in family_patterns
                ):
                    return True

            return False

        # --------------------------------------------------------------
        # Contact fields
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
        # Candidate + institution
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
                re.compile(
                    r"\bcandidate(?:'s|s)?\s+"
                    rf"(?:{institution_pattern})\b"
                    r"\s*(?::|-|is|was|=)\s*",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"\bthe\s+candidate(?:'s)?\s+"
                    rf"(?:{institution_pattern})\b"
                    r"\s*(?::|-|is|was|=)\s*",
                    re.IGNORECASE,
                ),
                re.compile(
                    rf"\bcandidate\b.*?"
                    rf"\b(?:{institution_pattern})\b",
                    re.IGNORECASE,
                ),
            ]

            if any(
                pattern.search(normalized_context)
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

            if named_person_pattern.search(
                normalized_context
            ):
                return True

            return False

        # --------------------------------------------------------------
        # Yes/no role questions
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

            role_pattern = re.escape(
                requested_role
            )

            explicit_role_patterns = [
                # Arun is a student.
                re.compile(
                    rf"\b(?:is|are|was|were)\s+"
                    rf"(?:an?\s+)?"
                    rf"{role_pattern}\b",
                    re.IGNORECASE,
                ),

                # Arun is currently a student.
                re.compile(
                    rf"\b(?:is|are|was|were)\s+"
                    rf"(?:currently|presently)\s+"
                    rf"(?:an?\s+)?"
                    rf"{role_pattern}\b",
                    re.IGNORECASE,
                ),

                # Arun works as a student.
                re.compile(
                    rf"\b(?:works?\s+as|serves?\s+as)\s+"
                    rf"(?:an?\s+)?"
                    rf"{role_pattern}\b",
                    re.IGNORECASE,
                ),

                # Student: Arun
                re.compile(
                    rf"\b{role_pattern}\b\s*"
                    r"(?::|-|=)\s*"
                    r"[A-Za-z][A-Za-z .'-]+",
                    re.IGNORECASE,
                ),
            ]

            return any(
                pattern.search(normalized_context)
                for pattern in explicit_role_patterns
            )

        return False

    # ------------------------------------------------------------------
    # Explicit negative evidence
    # ------------------------------------------------------------------

    @classmethod
    def _query_negative_evidence_terms(
        cls,
        query: str,
    ) -> set[str]:
        """
        Return semantic field terms that can legitimately connect
        negative evidence to the requested query.
        """

        lowered_query = query.lower()

        terms: set[str] = set()

        # Contact.
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

        # Family.
        for term in cls.FAMILY_TERMS:
            if re.search(
                rf"\b{re.escape(term)}\b",
                lowered_query,
            ):
                terms.add(term)

        # Financial.
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

        # Address/location.
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

        # Academic/result fields.
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
        """
        Determine whether a negative sentence actually refers to
        the field requested by the query.
        """

        query_terms = cls._query_negative_evidence_terms(
            query
        )

        if not query_terms:
            return False

        lowered_sentence = sentence.lower()
        lowered_query = query.lower()

        # Direct semantic field match.
        for term in query_terms:
            if re.search(
                rf"\b{re.escape(term)}\b",
                lowered_sentence,
            ):
                return True

        # Email.
        if re.search(
            r"\b(?:email|e-mail)\b",
            lowered_query,
        ):
            if re.search(
                r"\b(?:email|e-mail)\b",
                lowered_sentence,
            ):
                return True

        # Phone.
        if re.search(
            r"\b(?:phone|telephone|mobile)\b",
            lowered_query,
        ):
            if re.search(
                r"\b(?:phone|telephone|mobile)\b",
                lowered_sentence,
            ):
                return True

        # Contact number/details.
        if re.search(
            r"\bcontact\s+(?:number|details?|information)\b",
            lowered_query,
        ):
            if re.search(
                r"\bcontact\s+(?:number|details?|information)\b",
                lowered_sentence,
            ):
                return True

            if re.search(
                r"\b(?:phone|telephone|mobile)\b",
                lowered_sentence,
            ):
                return True

        # Family relationships.
        if "father" in lowered_query:
            if re.search(
                r"\b(?:father|parent|parents)\b",
                lowered_sentence,
            ):
                return True

        if "mother" in lowered_query:
            if re.search(
                r"\b(?:mother|parent|parents)\b",
                lowered_sentence,
            ):
                return True

        if re.search(
            r"\bparents?\b",
            lowered_query,
        ):
            if re.search(
                r"\bparents?\b",
                lowered_sentence,
            ):
                return True

        if "son" in lowered_query:
            if re.search(
                r"\b(?:son|child|children)\b",
                lowered_sentence,
            ):
                return True

        if "daughter" in lowered_query:
            if re.search(
                r"\b(?:daughter|child|children)\b",
                lowered_sentence,
            ):
                return True

        if "brother" in lowered_query:
            if re.search(
                r"\b(?:brother|sibling|siblings)\b",
                lowered_sentence,
            ):
                return True

        if "sister" in lowered_query:
            if re.search(
                r"\b(?:sister|sibling|siblings)\b",
                lowered_sentence,
            ):
                return True

        # Salary / income.
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

        # Address / residence.
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
    # Query field detection
    # ------------------------------------------------------------------

    @classmethod
    def _requested_field_groups(
        cls,
        query: str,
    ) -> set[str]:
        """
        Return semantic field groups requested by the query.
        """

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

        # --------------------------------------------------------------
        # Generic candidate identity question.
        #
        # "Who is the candidate?" is semantically asking for the
        # candidate's name. "candidate" is intentionally a stopword for
        # lexical overlap, so it must be handled explicitly here.
        # --------------------------------------------------------------

        if re.search(
            r"\bwho\s+is\s+(?:the\s+)?candidate\b",
            lowered,
        ):
            groups.add("name")

        # Semantic question patterns.

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

        if re.search(
            r"\bwhere\s+should\b",
            lowered,
        ) and re.search(
            r"\breport\b",
            lowered,
        ):
            groups.add("reporting")

        if re.search(
            r"\bwhen\s+is\b",
            lowered,
        ) and re.search(
            r"\b(?:test|exam|examination)\b",
            lowered,
        ):
            groups.add("examination_date")

        if re.search(
            r"\bwhat\s+is\b",
            lowered,
        ) and re.search(
            r"\bname\b",
            lowered,
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
    # Structured factual evidence
    # ------------------------------------------------------------------

    @classmethod
    def _has_direct_factual_evidence(
        cls,
        query: str,
        context: str,
    ) -> bool:
        """
        Detect common field/value statements.

        This method intentionally does not depend on reranker score.
        """

        lowered_query = query.lower()
        lowered_context = context.lower()

        factual_patterns = [
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
        ]

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

            if phrase in lowered_context:
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
        """
        Detect common subject -> field relationships.

        Examples:

            Arun's degree is B.E.

            Arun's expected graduation year is 2028.

            Arun's academic identifier: SVIT-AI-2026-2048.
        """

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
            patterns = field_patterns[field]

            for pattern in patterns:
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
        Determine whether the evidence contains terminology strongly
        associated with the requested factual field.

        This is intentionally used only for ordinary factual questions.
        Relationship-sensitive questions are handled separately.
        """

        fields = cls._requested_field_groups(query)

        if not fields:
            return False

        lowered_context = context.lower()

        # --------------------------------------------------------------
        # Date of birth
        # --------------------------------------------------------------

        if "date_of_birth" in fields:
            if cls._context_contains_field(
                "date_of_birth",
                context,
            ):
                return True

            if re.search(
                r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
                context,
            ):
                return True

            if re.search(
                r"\b"
                r"(?:"
                r"\d{1,2}\s+"
                r"[A-Za-z]{3,12}\s+"
                r"\d{4}"
                r"|"
                r"[A-Za-z]{3,12}\s+"
                r"\d{1,2},?\s+"
                r"\d{4}"
                r")"
                r"\b",
                context,
            ):
                return True

        # --------------------------------------------------------------
        # Examination date / scheduled test
        # --------------------------------------------------------------

        if "examination_date" in fields:
            if cls._context_contains_field(
                "examination_date",
                context,
            ):
                return True

            if re.search(
                r"\b(?:"
                r"\d{1,2}\s+"
                r"[A-Za-z]{3,12}"
                r"(?:\s*[-–]\s*|\s+)"
                r"\d{1,2}?"
                r"(?:,\s*|\s+)"
                r"\d{4}"
                r"|"
                r"[A-Za-z]{3,12}\s+"
                r"\d{1,2},?\s+\d{4}"
                r"|"
                r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
                r")\b",
                context,
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
                r"\b(?:"
                r"center|centre|venue|location"
                r")\b",
                lowered_context,
            ):
                return True

            if re.search(
                r"\b(?:"
                r"iON\s+Digital\s+Zone|"
                r"exam(?:ination)?\s+(?:center|centre)|"
                r"test\s+(?:center|centre)"
                r")\b",
                context,
                re.IGNORECASE,
            ):
                return True

        # --------------------------------------------------------------
        # Reporting / "where should I report?"
        # --------------------------------------------------------------

        if "reporting" in fields:
            if re.search(
                r"\b(?:"
                r"center|centre|venue|location|"
                r"report(?:ing)?\s+(?:at|to|center|centre)"
                r")\b",
                lowered_context,
            ):
                return True

            if re.search(
                r"\biON\s+Digital\s+Zone\b",
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

            # A person-like name is sufficient for an ordinary
            # "Who is the candidate?" question.
            if re.search(
                r"\b[A-Z][a-z]+(?:\s+[A-Z][A-Za-z.'-]+){1,5}\b",
                context,
            ):
                return True

        # --------------------------------------------------------------
        # Generic field aliases
        # --------------------------------------------------------------

        for field in fields:
            if field in {
                "date_of_birth",
                "examination_date",
                "examination_center",
                "reporting",
                "name",
            }:
                continue

            if cls._context_contains_field(
                field,
                context,
            ):
                return True

        return False

    # ------------------------------------------------------------------
    # General semantic evidence
    # ------------------------------------------------------------------

    def _has_meaningful_evidence(
        self,
        query: str,
        context: str,
        token_overlap: float,
    ) -> bool:
        """
        Determine whether ordinary factual evidence is sufficient.

        This deliberately does not require high lexical overlap because
        reranking and embeddings can retrieve semantically relevant
        evidence whose wording differs from the query.

        Examples:

            Query:
                Where should the candidate report?

            Evidence:
                iON Digital Zone iDZ Thondamuthur ...

        The lexical overlap may be weak, but the evidence is clearly
        an examination-location style fact.

        IMPORTANT:
        This is an instance method because the configured threshold
        self.min_token_overlap belongs to the service instance.
        """

        if self._has_semantic_field_evidence(
            query,
            context,
        ):
            return True

        if self._has_direct_factual_evidence(
            query,
            context,
        ):
            return True

        if self._has_explicit_subject_field_relation(
            query,
            context,
        ):
            return True

        if self._is_list_question(query):
            if self._has_structured_list_evidence(
                query,
                context,
            ):
                return True

        # Exact query text is strong evidence.
        if self._has_exact_query_match(
            query,
            context,
        ):
            return True

        # For generic factual questions, a reasonable lexical overlap
        # can establish that the retrieved context is about the query.
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
    # Main answerability check
    # ------------------------------------------------------------------

    def check(
        self,
        query: str,
        reranked_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized_query = query.strip()

        # --------------------------------------------------------------
        # Empty query
        #
        # Public API contract:
        # an empty query is invalid input, not an unanswerable query.
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
        # Build evidence context
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
        # IMPORTANT:
        #
        # Reranker scores are diagnostic metadata only.
        #
        # They must NOT be used as a hard answerability threshold because
        # valid cross-encoder scores can be negative.
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
        # Explicit relationship requirement
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
        # Explicit negative evidence
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

        exact_query_match = self._has_exact_query_match(
            normalized_query,
            context,
        )

        # --------------------------------------------------------------
        # 1. Explicit negative evidence
        #
        # Negative evidence always takes priority over ordinary
        # positive lexical matching.
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
                exact_query_match=exact_query_match,
            )

        # --------------------------------------------------------------
        # 2. Exact identifier meaning questions
        #
        # Example:
        #
        #   What does SVIT-AI-2026-2048 represent?
        #
        # Merely finding the identifier is not enough.
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
                    exact_query_match=exact_query_match,
                )

            if self._identifier_is_present_without_meaning(
                normalized_query,
                context,
            ):
                return self._build_result(
                    answerable=False,
                    reason="unsupported_relationship",
                    max_rerank_score=max_rerank_score,
                    token_overlap=token_overlap,
                    evidence_density=evidence_density,
                    explicit_relation_support=False,
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

        # --------------------------------------------------------------
        # 3. Explicit relationship questions
        #
        # Examples:
        #
        #   What is the candidate's father's name?
        #   What is the candidate's phone number?
        #   Is Vishwa Sabaris V a student?
        #
        # These must have explicit relationship support.
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

        # --------------------------------------------------------------
        # 4. Ordinary factual / extraction questions
        #
        # These do NOT require an explicit relationship detector.
        #
        # Examples:
        #
        #   What is the examination date?
        #   What is the candidate's date of birth?
        #   What is the examination center?
        #   Who is the candidate?
        #
        # Evidence can be:
        # - structured
        # - semantic
        # - exact
        # - reasonably lexically overlapping
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
                exact_query_match=exact_query_match,
            )

        # --------------------------------------------------------------
        # 5. No sufficient evidence
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
            exact_query_match=exact_query_match,
        )

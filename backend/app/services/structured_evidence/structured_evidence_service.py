from __future__ import annotations

import re
from typing import Any


class StructuredEvidenceService:
    """
    Deterministic evidence extraction for semi-structured documents.

    The service is deliberately conservative.

    It extracts values only when the document provides enough
    structural evidence to connect the value with the requested field.

    It does NOT:
    - use an LLM
    - use outside knowledge
    - infer relationships from simple proximity
    - decide what an arbitrary identifier means
    """

    NOT_AVAILABLE_MESSAGE = (
        "The information is not available in the provided document."
    )

    # -------------------------------------------------------------
    # Field aliases
    # -------------------------------------------------------------

    SUBJECT_TERMS = (
        "subject",
        "course",
        "course title",
        "subject name",
        "course name",
    )

    REGISTRATION_TERMS = (
        "registration number",
        "registration no",
        "registration num",
        "registration id",
        "registration identifier",
        "registration",
    )

    EXAMINATION_DATE_TERMS = (
        "examination date",
        "exam date",
        "examination day",
        "exam day",
        "test date",
        "test day",
    )

    EXAMINATION_CENTER_TERMS = (
        "examination center",
        "examination centre",
        "exam center",
        "exam centre",
        "examination venue",
        "exam venue",
        "test center",
        "test centre",
        "test venue",
    )

    POSTAL_CODE_TERMS = (
        "postal code",
        "postcode",
        "pin code",
        "pincode",
        "zip code",
        "zip",
    )

    DATE_OF_BIRTH_TERMS = (
        "date of birth",
        "dob",
        "birth date",
        "birthdate",
    )

    COLLEGE_TERMS = (
        "college",
        "university",
        "school",
        "institute",
        "institution",
    )

    # -------------------------------------------------------------
    # Identifier patterns
    # -------------------------------------------------------------

    # Example:
    #
    # NOC26CS117S353802605
    #
    # This is deliberately generic:
    # - must contain letters
    # - must contain digits
    # - allows common identifier punctuation
    #
    IDENTIFIER_PATTERN = re.compile(
        r"(?<![A-Za-z0-9._:/-])"
        r"(?=[A-Za-z0-9._:/-]*[A-Za-z])"
        r"(?=[A-Za-z0-9._:/-]*\d)"
        r"[A-Za-z0-9._:/-]{6,}"
        r"(?![A-Za-z0-9._:/-])"
    )

    # NPTEL-style registration identifiers often look like:
    #
    # NOC26CS117S353802605
    #
    # This pattern is intentionally not tied to one exact value.
    REGISTRATION_IDENTIFIER_PATTERN = re.compile(
        r"\b"
        r"[A-Za-z]{2,}\d+"
        r"[A-Za-z0-9-]*"
        r"\b"
    )

    # Short numeric values can also appear as a registration number,
    # but we only accept them when they are structurally attached to
    # a registration label.
    SHORT_NUMERIC_IDENTIFIER_PATTERN = re.compile(
        r"\b\d{4,12}\b"
    )

    # -------------------------------------------------------------
    # Month names
    # -------------------------------------------------------------

    MONTH_PATTERN = (
        r"(?:"
        r"Jan|January|"
        r"Feb|February|"
        r"Mar|March|"
        r"Apr|April|"
        r"May|"
        r"Jun|June|"
        r"Jul|July|"
        r"Aug|August|"
        r"Sep|September|"
        r"Oct|October|"
        r"Nov|November|"
        r"Dec|December"
        r")"
    )

    # -------------------------------------------------------------
    # General helpers
    # -------------------------------------------------------------

    @staticmethod
    def _normalize_whitespace(
        text: str,
    ) -> str:
        return " ".join(
            text.strip().split()
        )

    @classmethod
    def _normalize(
        cls,
        text: str,
    ) -> str:
        return cls._normalize_whitespace(
            text
        ).lower()

    @classmethod
    def _query_has_term(
        cls,
        query: str,
        terms: tuple[str, ...],
    ) -> str | None:
        lowered = cls._normalize(query)

        for term in sorted(
            terms,
            key=len,
            reverse=True,
        ):
            if re.search(
                rf"\b{re.escape(term)}\b",
                lowered,
            ):
                return term

        return None

    @classmethod
    def _find_source_chunk(
        cls,
        value: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        normalized_value = cls._normalize(
            value
        )

        for chunk in chunks:
            text = cls._normalize(
                str(
                    chunk.get(
                        "text",
                        "",
                    )
                )
            )

            if normalized_value in text:
                return chunk

        return chunks[0] if chunks else None

    # -------------------------------------------------------------
    # Generic labeled-value extraction
    # -------------------------------------------------------------

    @classmethod
    def _find_labeled_value(
        cls,
        text: str,
        labels: tuple[str, ...],
    ) -> str | None:
        """
        Find explicit label/value structures.

        Examples:

            Subject: CS117 Data Base Management System

            Registration Number: NOC26CS117S353802605

            Date of Birth: 28-07-2008

            Examination Center: iON Digital Zone
        """

        for label in sorted(
            labels,
            key=len,
            reverse=True,
        ):
            pattern = re.compile(
                rf"(?im)"
                rf"^\s*"
                rf"{re.escape(label)}"
                rf"\s*(?::|-|=|is|was)\s*"
                rf"([^\n\r]+)"
            )

            match = pattern.search(
                text
            )

            if match:
                value = match.group(1).strip()

                if value:
                    return value

        return None

    @classmethod
    def _find_inline_label_value(
        cls,
        text: str,
        labels: tuple[str, ...],
    ) -> str | None:
        """
        Handle inline structures.

        Examples:

            Subject - CS117 Data Base Management System

            Registration Number = NOC26CS117S353802605

            Postal Code = 641109
        """

        normalized_text = cls._normalize_whitespace(
            text
        )

        for label in sorted(
            labels,
            key=len,
            reverse=True,
        ):
            pattern = re.compile(
                rf"\b{re.escape(label)}\b"
                rf"\s*(?::|-|=|is|was)\s*"
                rf"([^|;\n]+)",
                re.IGNORECASE,
            )

            match = pattern.search(
                normalized_text
            )

            if match:
                value = match.group(1).strip()

                if value:
                    return value

        return None

    # -------------------------------------------------------------
    # Subject extraction
    # -------------------------------------------------------------

    @classmethod
    def _extract_subject_from_exam_header(
        cls,
        text: str,
    ) -> str | None:
        """
        Detect an NPTEL-style examination header.

        Example:

            SEM2NOC26: CS117 Data Base Management System 20 Sep - Online

        Returns:

            CS117 Data Base Management System
        """

        patterns = [
            re.compile(
                r"\b[A-Z0-9]+:\s*"
                r"(CS\d+)\s+"
                r"(.+?)"
                rf"\s+\d{{1,2}}\s+"
                rf"{cls.MONTH_PATTERN}\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b"
                r"(CS\d+)\s+"
                r"(.+?)"
                rf"\s+\d{{1,2}}\s+"
                rf"{cls.MONTH_PATTERN}\b",
                re.IGNORECASE,
            ),
        ]

        for pattern in patterns:
            match = pattern.search(
                text
            )

            if not match:
                continue

            code = match.group(1).strip()
            title = match.group(2).strip()

            title = re.sub(
                r"\s*[-–—]\s*$",
                "",
                title,
            ).strip()

            if code and title:
                return f"{code} {title}"

        return None

    @classmethod
    def _extract_subject(
        cls,
        text: str,
    ) -> str | None:
        value = cls._find_labeled_value(
            text,
            cls.SUBJECT_TERMS,
        )

        if value:
            return value

        value = cls._find_inline_label_value(
            text,
            cls.SUBJECT_TERMS,
        )

        if value:
            return value

        return cls._extract_subject_from_exam_header(
            text
        )

    # -------------------------------------------------------------
    # Registration number extraction
    # -------------------------------------------------------------

    @classmethod
    def _clean_registration_value(
        cls,
        value: str,
    ) -> str | None:
        """
        Extract the actual identifier from a labeled value.

        Examples:

            Registration Number: NOC26CS117S353802605

            Registration No: NOC26CS117S353802605

            Registration Number: 2605
        """

        value = value.strip()

        if not value:
            return None

        # Prefer an alphanumeric identifier containing letters + digits.
        match = cls.REGISTRATION_IDENTIFIER_PATTERN.search(
            value
        )

        if match:
            candidate = match.group(0).strip()

            if (
                re.search(r"[A-Za-z]", candidate)
                and re.search(r"\d", candidate)
            ):
                return candidate

        # Otherwise accept an explicitly labeled numeric value.
        match = cls.SHORT_NUMERIC_IDENTIFIER_PATTERN.search(
            value
        )

        if match:
            return match.group(0)

        return None

    @classmethod
    def _extract_registration_number(
        cls,
        text: str,
    ) -> str | None:
        """
        Extract registration number only when it has explicit
        registration-field support.

        We intentionally do NOT treat:

            NOC26CS117S353802605 2605

        as proof that 2605 is the registration number.

        The complete identifier can be returned when the document
        explicitly labels it as the registration number.
        """

        value = cls._find_labeled_value(
            text,
            cls.REGISTRATION_TERMS,
        )

        if value:
            extracted = cls._clean_registration_value(
                value
            )

            if extracted:
                return extracted

        value = cls._find_inline_label_value(
            text,
            cls.REGISTRATION_TERMS,
        )

        if value:
            extracted = cls._clean_registration_value(
                value
            )

            if extracted:
                return extracted

        return None

    # -------------------------------------------------------------
    # Examination date extraction
    # -------------------------------------------------------------

    @classmethod
    def _extract_date_from_header(
        cls,
        text: str,
    ) -> str | None:
        """
        Detect examination dates from common exam-header structures.

        Examples:

            NPTEL EXAM – 20 September, 2026

            Sunday, 20 September, 2026

            CS117 Data Base Management System 20 Sep
        """

        patterns = [
            re.compile(
                rf"\b"
                rf"\d{{1,2}}\s+"
                rf"{cls.MONTH_PATTERN}"
                rf"(?:,\s*|\s+)"
                rf"\d{{4}}"
                rf"\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\b"
                rf"{cls.MONTH_PATTERN}\s+"
                rf"\d{{1,2}}"
                rf",?\s+"
                rf"\d{{4}}"
                rf"\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b"
                r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
                r"\b"
            ),
        ]

        for pattern in patterns:
            match = pattern.search(text)

            if match:
                return match.group(0).strip()

        return None

    @classmethod
    def _extract_examination_date(
        cls,
        text: str,
    ) -> str | None:
        """
        Prefer explicit examination-date labels.

        If there is no label, allow a date from an explicit exam
        header such as:

            NPTEL EXAM – 20 September, 2026
        """

        value = cls._find_labeled_value(
            text,
            cls.EXAMINATION_DATE_TERMS,
        )

        if value:
            date_match = re.search(
                rf"\b(?:"
                rf"\d{{1,2}}\s+{cls.MONTH_PATTERN}"
                rf"(?:,\s*|\s+)\d{{4}}"
                rf"|"
                rf"{cls.MONTH_PATTERN}\s+\d{{1,2}},?\s+\d{{4}}"
                rf"|"
                rf"\d{{1,2}}[-/]\d{{1,2}}[-/]\d{{2,4}}"
                rf")\b",
                value,
                re.IGNORECASE,
            )

            if date_match:
                return date_match.group(0).strip()

        value = cls._find_inline_label_value(
            text,
            cls.EXAMINATION_DATE_TERMS,
        )

        if value:
            date_match = re.search(
                rf"\b(?:"
                rf"\d{{1,2}}\s+{cls.MONTH_PATTERN}"
                rf"(?:,\s*|\s+)\d{{4}}"
                rf"|"
                rf"{cls.MONTH_PATTERN}\s+\d{{1,2}},?\s+\d{{4}}"
                rf"|"
                rf"\d{{1,2}}[-/]\d{{1,2}}[-/]\d{{2,4}}"
                rf")\b",
                value,
                re.IGNORECASE,
            )

            if date_match:
                return date_match.group(0).strip()

        # Header fallback is only used for an examination-date query.
        return cls._extract_date_from_header(
            text
        )

    # -------------------------------------------------------------
    # Examination center extraction
    # -------------------------------------------------------------

    @classmethod
    def _extract_examination_center(
        cls,
        text: str,
    ) -> str | None:
        """
        Extract an examination center/venue.

        Explicit examples:

            Examination Center: iON Digital Zone iDZ Thondamuthur

            Exam Centre: ABC College

        For the common NPTEL hall-ticket layout, the following
        structure is also supported:

            iON Digital Zone iDZ Thondamuthur
            Ranganathan Engineering College, REC Kalvi Nagar,
            Viraliyur Post, Thondamuthur via Coimbatore,
            Tamil Nadu, India 641109.

        In that layout, the first line is the examination venue.
        """

        value = cls._find_labeled_value(
            text,
            cls.EXAMINATION_CENTER_TERMS,
        )

        if value:
            return value

        value = cls._find_inline_label_value(
            text,
            cls.EXAMINATION_CENTER_TERMS,
        )

        if value:
            return value

        # ---------------------------------------------------------
        # Common NPTEL/iON venue structure
        # ---------------------------------------------------------

        venue_pattern = re.compile(
            r"(?im)"
            r"^\s*"
            r"(iON\s+Digital\s+Zone"
            r"(?:\s+[^\n\r]+)?)"
            r"\s*$"
        )

        match = venue_pattern.search(
            text
        )

        if match:
            value = match.group(1).strip()

            # Remove accidental trailing punctuation.
            value = value.rstrip(".,;:-")

            if value:
                return value

        # ---------------------------------------------------------
        # Generic venue/center line
        # ---------------------------------------------------------

        generic_pattern = re.compile(
            r"(?im)"
            r"^\s*"
            r"((?:Exam(?:ination)?|Test)"
            r"\s+(?:Center|Centre|Venue)"
            r"\s*(?::|-|=)\s*"
            r"[^\n\r]+)"
        )

        match = generic_pattern.search(
            text
        )

        if match:
            value = match.group(1).strip()

            value = re.sub(
                r"^(?:Exam(?:ination)?|Test)"
                r"\s+(?:Center|Centre|Venue)"
                r"\s*(?::|-|=)\s*",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip()

            if value:
                return value

        return None

    # -------------------------------------------------------------
    # Postal code extraction
    # -------------------------------------------------------------

    @classmethod
    def _extract_postal_code(
        cls,
        text: str,
    ) -> str | None:
        # Explicit postal-code label.
        value = cls._find_labeled_value(
            text,
            cls.POSTAL_CODE_TERMS,
        )

        if value:
            match = re.search(
                r"\b\d{5,6}\b",
                value,
            )

            if match:
                return match.group(0)

        value = cls._find_inline_label_value(
            text,
            cls.POSTAL_CODE_TERMS,
        )

        if value:
            match = re.search(
                r"\b\d{5,6}\b",
                value,
            )

            if match:
                return match.group(0)

        # ---------------------------------------------------------
        # Address-line extraction
        # ---------------------------------------------------------
        #
        # Example:
        #
        # via Coimbatore, Tamil Nadu, India 641109
        #
        # A six-digit number at the end of an address line is
        # accepted because the address structure supports it.
        # ---------------------------------------------------------

        address_pattern = re.compile(
            r"(?im)"
            r"^\s*"
            r".*?"
            r",\s*"
            r"(?:India|INDIA)"
            r"\s*,?\s*"
            r"(\d{6})"
            r"\s*[.]?\s*$"
        )

        match = address_pattern.search(
            text
        )

        if match:
            return match.group(1)

        general_address_pattern = re.compile(
            r"(?im)"
            r"^\s*"
            r".{5,}"
            r",\s*"
            r"(\d{6})"
            r"\s*[.]?\s*$"
        )

        match = general_address_pattern.search(
            text
        )

        if match:
            return match.group(1)

        return None

    # -------------------------------------------------------------
    # Date of birth extraction
    # -------------------------------------------------------------

    @classmethod
    def _extract_date_of_birth(
        cls,
        text: str,
    ) -> str | None:
        """
        Only return a DOB when explicitly labeled.

        Therefore:

            28-07-2008

        by itself is NOT treated as a DOB.
        """

        value = cls._find_labeled_value(
            text,
            cls.DATE_OF_BIRTH_TERMS,
        )

        if value:
            return value

        value = cls._find_inline_label_value(
            text,
            cls.DATE_OF_BIRTH_TERMS,
        )

        return value

    # -------------------------------------------------------------
    # College extraction
    # -------------------------------------------------------------

    @classmethod
    def _extract_person_from_query(
        cls,
        query: str,
    ) -> str | None:
        """
        Extract a person from:

            What is Vishwa Sabaris V's college?

        Returns:

            Vishwa Sabaris V
        """

        pattern = re.compile(
            r"^\s*what\s+is\s+"
            r"(.+?)"
            r"['’]s\s+"
            r".+?\??\s*$",
            re.IGNORECASE,
        )

        match = pattern.match(
            query.strip()
        )

        if not match:
            return None

        person = match.group(1).strip()

        return person or None

    @classmethod
    def _extract_explicit_college(
        cls,
        text: str,
        person: str | None,
    ) -> str | None:
        """
        Extract a college only when the person → college relationship
        is explicitly stated.

        Accepted examples:

            Vishwa Sabaris V's college: ABC College

            Vishwa Sabaris V's college is ABC College
        """

        if not person:
            return None

        normalized_person = cls._normalize(
            person
        )

        escaped_person = re.escape(
            normalized_person
        )

        college_terms = "|".join(
            re.escape(term)
            for term in sorted(
                cls.COLLEGE_TERMS,
                key=len,
                reverse=True,
            )
        )

        patterns = [
            re.compile(
                rf"\b{escaped_person}\b"
                rf"\s*['’]s\s+"
                rf"(?:{college_terms})\b"
                rf"\s*(?::|-|is|was|=)\s*"
                rf"(?P<value>[^\n\r]+)",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\b{escaped_person}\b"
                rf".{{0,80}}?"
                rf"\b(?:{college_terms})\b"
                rf"\s*(?::|-|=)\s*"
                rf"(?P<value>[^\n\r]+)",
                re.IGNORECASE,
            ),
        ]

        normalized_text = cls._normalize_whitespace(
            text
        )

        for pattern in patterns:
            match = pattern.search(
                normalized_text
            )

            if match:
                value = match.group(
                    "value"
                ).strip()

                if value:
                    return value

        return None

    # -------------------------------------------------------------
    # Main API
    # -------------------------------------------------------------

    @classmethod
    def extract(
        cls,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Return deterministic evidence when the requested field has
        sufficient structural support.

        Returns:

            {
                "value": "...",
                "field": "...",
                "chunk": {...},
                "reason": "..."
            }

        Otherwise:

            None
        """

        if not query.strip():
            return None

        if not chunks:
            return None

        combined_text = "\n".join(
            str(
                chunk.get(
                    "text",
                    "",
                )
            ).strip()
            for chunk in chunks
            if str(
                chunk.get(
                    "text",
                    "",
                )
            ).strip()
        )

        if not combined_text:
            return None

        # ---------------------------------------------------------
        # Subject
        # ---------------------------------------------------------

        if cls._query_has_term(
            query,
            cls.SUBJECT_TERMS,
        ):
            value = cls._extract_subject(
                combined_text
            )

            if value:
                source = cls._find_source_chunk(
                    value=value,
                    chunks=chunks,
                )

                if source:
                    return {
                        "value": value,
                        "field": "subject",
                        "chunk": source,
                        "reason": (
                            "explicit_subject_evidence"
                        ),
                    }

        # ---------------------------------------------------------
        # Registration number
        # ---------------------------------------------------------

        if cls._query_has_term(
            query,
            cls.REGISTRATION_TERMS,
        ):
            value = cls._extract_registration_number(
                combined_text
            )

            if value:
                source = cls._find_source_chunk(
                    value=value,
                    chunks=chunks,
                )

                if source:
                    return {
                        "value": value,
                        "field": "registration_number",
                        "chunk": source,
                        "reason": (
                            "explicit_registration_number_evidence"
                        ),
                    }

        # ---------------------------------------------------------
        # Examination date
        # ---------------------------------------------------------

        if cls._query_has_term(
            query,
            cls.EXAMINATION_DATE_TERMS,
        ):
            value = cls._extract_examination_date(
                combined_text
            )

            if value:
                source = cls._find_source_chunk(
                    value=value,
                    chunks=chunks,
                )

                if source:
                    return {
                        "value": value,
                        "field": "examination_date",
                        "chunk": source,
                        "reason": (
                            "explicit_examination_date_evidence"
                        ),
                    }

        # ---------------------------------------------------------
        # Examination center
        # ---------------------------------------------------------

        if cls._query_has_term(
            query,
            cls.EXAMINATION_CENTER_TERMS,
        ):
            value = cls._extract_examination_center(
                combined_text
            )

            if value:
                source = cls._find_source_chunk(
                    value=value,
                    chunks=chunks,
                )

                if source:
                    return {
                        "value": value,
                        "field": "examination_center",
                        "chunk": source,
                        "reason": (
                            "explicit_examination_center_evidence"
                        ),
                    }

        # ---------------------------------------------------------
        # Postal code
        # ---------------------------------------------------------

        if cls._query_has_term(
            query,
            cls.POSTAL_CODE_TERMS,
        ):
            value = cls._extract_postal_code(
                combined_text
            )

            if value:
                source = cls._find_source_chunk(
                    value=value,
                    chunks=chunks,
                )

                if source:
                    return {
                        "value": value,
                        "field": "postal_code",
                        "chunk": source,
                        "reason": (
                            "explicit_address_postal_evidence"
                        ),
                    }

        # ---------------------------------------------------------
        # Date of birth
        # ---------------------------------------------------------

        if cls._query_has_term(
            query,
            cls.DATE_OF_BIRTH_TERMS,
        ):
            value = cls._extract_date_of_birth(
                combined_text
            )

            if value:
                source = cls._find_source_chunk(
                    value=value,
                    chunks=chunks,
                )

                if source:
                    return {
                        "value": value,
                        "field": "date_of_birth",
                        "chunk": source,
                        "reason": (
                            "explicit_date_of_birth_label"
                        ),
                    }

            # Do NOT fall back to arbitrary dates.
            return None

        # ---------------------------------------------------------
        # College
        # ---------------------------------------------------------

        if cls._query_has_term(
            query,
            cls.COLLEGE_TERMS,
        ):
            person = cls._extract_person_from_query(
                query
            )

            value = cls._extract_explicit_college(
                text=combined_text,
                person=person,
            )

            if value:
                source = cls._find_source_chunk(
                    value=value,
                    chunks=chunks,
                )

                if source:
                    return {
                        "value": value,
                        "field": "college",
                        "chunk": source,
                        "reason": (
                            "explicit_person_college_relation"
                        ),
                    }

            # Merely mentioning a college is not enough.
            return None

        return None

from __future__ import annotations

import re
from typing import Any


class StructuredEvidenceService:
    """
    Deterministic structured-evidence extraction for DocuRAG.

    Design principles
    -----------------
    1. Extract literal values only when the document provides
       structural evidence for the requested field.

    2. Never infer relationships merely from:
       - textual proximity
       - visual proximity
       - the order of fields
       - a name appearing near a value
       - a number appearing near a label
       - an institution appearing near a person's name

    3. Explicit field/value structures are accepted.

    4. Person-specific questions require an explicit relationship
       between the requested person/entity and the requested field.

    5. The implementation is document-agnostic. It is not tied to
       Hall Tickets or any particular PDF layout.

    6. No LLM or external knowledge is used.
    """

    NOT_AVAILABLE_MESSAGE = (
        "The information is not available in the provided document."
    )

    # ------------------------------------------------------------------
    # Field aliases
    # ------------------------------------------------------------------

    SUBJECT_TERMS = (
        "subject",
        "course",
        "course title",
        "subject name",
        "course name",
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

    EXAMINATION_DATE_TERMS = (
        "examination date",
        "exam date",
        "date of examination",
        "date of exam",
        "test date",
    )

    EXAMINATION_CENTER_TERMS = (
        "examination center",
        "examination centre",
        "exam center",
        "exam centre",
        "test center",
        "test centre",
        "exam venue",
        "examination venue",
        "test venue",
    )

    REGISTRATION_NUMBER_TERMS = (
        "registration number",
        "registration no",
        "registration id",
        "registration code",
        "registration",
    )

    IDENTIFIER_TERMS = (
        "identifier",
        "id",
        "student id",
        "candidate id",
        "employee id",
        "application id",
        "application number",
        "reference number",
        "reference id",
        "registration number",
        "registration no",
        "registration id",
        "roll number",
        "roll no",
        "admission number",
        "admission no",
    )

    NAME_TERMS = (
        "name",
        "full name",
        "candidate name",
        "student name",
        "applicant name",
        "employee name",
        "person name",
    )

    DEGREE_TERMS = (
        "degree",
        "qualification",
        "academic qualification",
    )

    GRADUATION_TERMS = (
        "graduation",
        "graduation year",
        "expected graduation",
        "expected graduation year",
    )

    PHONE_TERMS = (
        "phone",
        "phone number",
        "telephone",
        "telephone number",
        "mobile",
        "mobile number",
        "contact number",
    )

    EMAIL_TERMS = (
        "email",
        "email address",
        "e-mail",
        "e-mail address",
    )

    ADDRESS_TERMS = (
        "address",
        "home address",
        "residential address",
        "street address",
        "location",
    )

    RESULT_TERMS = (
        "result",
        "score",
        "marks",
        "mark",
        "grade",
        "rank",
        "percentage",
        "percentile",
        "cgpa",
        "gpa",
        "classification",
        "performance",
    )

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return " ".join(text.strip().split())

    @classmethod
    def _normalize(cls, text: str) -> str:
        return cls._normalize_whitespace(text).lower()

    @classmethod
    def _lines(cls, text: str) -> list[str]:
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

    @classmethod
    def _sentences(cls, text: str) -> list[str]:
        normalized = cls._normalize_whitespace(text)

        if not normalized:
            return []

        parts = re.split(
            r"(?<=[.!?])\s+",
            normalized,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

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
    def _query_has_any_term(
        cls,
        query: str,
        terms: tuple[str, ...],
    ) -> bool:
        return (
            cls._query_has_term(
                query,
                terms,
            )
            is not None
        )

    @classmethod
    def _extract_person_from_query(
        cls,
        query: str,
    ) -> str | None:
        """
        Extract a named subject from possessive questions.

        Examples:

            What is Vishwa Sabaris V's college?
            -> Vishwa Sabaris V

            What is John Smith's phone number?
            -> John Smith

            What is the candidate's date of birth?
            -> candidate

        Generic role words such as "candidate" are deliberately
        excluded from person-specific matching because they do not
        identify a concrete person.
        """

        pattern = re.compile(
            r"^\s*"
            r"(?:what|who|where|when|which)\s+"
            r"(?:is|are|was|were|has|have)\s+"
            r"(.+?)"
            r"['’]s\s+"
            r".+?\??"
            r"\s*$",
            re.IGNORECASE,
        )

        match = pattern.match(query.strip())

        if not match:
            return None

        person = match.group(1).strip()

        if not person:
            return None

        generic_subjects = {
            "candidate",
            "student",
            "applicant",
            "employee",
            "person",
            "user",
            "customer",
            "patient",
            "applicant",
            "the candidate",
            "the student",
            "the applicant",
            "the employee",
        }

        if cls._normalize(person) in generic_subjects:
            return None

        return person

    @classmethod
    def _query_targets_named_person(
        cls,
        query: str,
    ) -> bool:
        """
        Determine whether the question explicitly asks about a
        named entity/person using possessive syntax.

        Examples:

            Vishwa Sabaris V's college -> True
            John Smith's phone number -> True
            the candidate's college -> False
            candidate's date of birth -> False
        """

        return (
            cls._extract_person_from_query(query)
            is not None
        )

    @classmethod
    def _query_targets_generic_subject(
        cls,
        query: str,
    ) -> bool:
        lowered = cls._normalize(query)

        return bool(
            re.search(
                r"\b(?:candidate|student|applicant|employee|"
                r"person|user|customer|patient)\b"
                r"\s*['’]s\b",
                lowered,
            )
        )

    # ------------------------------------------------------------------
    # Generic label/value extraction
    # ------------------------------------------------------------------

    @classmethod
    def _find_labeled_value(
        cls,
        text: str,
        labels: tuple[str, ...],
    ) -> str | None:
        """
        Extract a value from an explicit label/value structure.

        Supported forms:

            Subject: Computer Science
            Date of Birth: 28-07-2008
            College: ABC College
            Email: user@example.com
            Registration Number: ABC123

        The value terminates at a newline.
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
                rf"\s*(?::|-|=)\s*"
                rf"([^\n\r]+)"
            )

            match = pattern.search(text)

            if not match:
                continue

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
        Handle inline structures such as:

            Subject - Computer Science
            College: ABC College | Degree: B.E.
            Email = user@example.com
        """

        normalized_text = cls._normalize_whitespace(text)

        for label in sorted(
            labels,
            key=len,
            reverse=True,
        ):
            pattern = re.compile(
                rf"\b{re.escape(label)}\b"
                rf"\s*(?::|-|=)\s*"
                rf"([^|;\n]+)",
                re.IGNORECASE,
            )

            match = pattern.search(normalized_text)

            if not match:
                continue

            value = match.group(1).strip()

            if value:
                return value

        return None

    @classmethod
    def _find_label_value_in_line(
        cls,
        line: str,
        labels: tuple[str, ...],
    ) -> str | None:
        for label in sorted(
            labels,
            key=len,
            reverse=True,
        ):
            pattern = re.compile(
                rf"^\s*"
                rf"{re.escape(label)}"
                rf"\s*(?::|-|=)\s*"
                rf"(.+?)"
                rf"\s*$",
                re.IGNORECASE,
            )

            match = pattern.match(line)

            if match:
                value = match.group(1).strip()

                if value:
                    return value

        return None

    # ------------------------------------------------------------------
    # Subject/course extraction
    # ------------------------------------------------------------------

    @classmethod
    def _extract_subject_from_exam_header(
        cls,
        text: str,
    ) -> str | None:
        """
        Generic exam-header support.

        This is deliberately structural rather than document-specific:
        a code followed by a title followed by a date can identify a
        course/subject when the question asks for the subject.

        It does not establish a person relationship.
        """

        patterns = [
            re.compile(
                r"\b[A-Z0-9_-]+:\s*"
                r"([A-Z]{1,10}\d{1,6})\s+"
                r"(.+?)"
                r"\s+\d{1,2}\s+"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
                r"|January|February|March|April|May|June|July|August|"
                r"September|October|November|December)"
                r"\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b"
                r"([A-Z]{1,10}\d{1,6})\s+"
                r"(.+?)"
                r"\s+\d{1,2}\s+"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
                r"|January|February|March|April|May|June|July|August|"
                r"September|October|November|December)"
                r"\b",
                re.IGNORECASE,
            ),
        ]

        for pattern in patterns:
            match = pattern.search(text)

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

    # ------------------------------------------------------------------
    # Postal code
    # ------------------------------------------------------------------

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

        # Address-structure fallback.
        #
        # A postal code can be inferred from a clearly structured
        # address ending in a postal code. This does NOT treat an
        # arbitrary number as a postal code.
        address_patterns = (
            re.compile(
                r"(?im)"
                r"^\s*"
                r".+?,\s*"
                r"(?:India|INDIA)"
                r"\s*,?\s*"
                r"(\d{6})"
                r"\s*[.]?\s*$"
            ),
            re.compile(
                r"(?im)"
                r"^\s*"
                r".{5,}"
                r",\s*"
                r"(\d{6})"
                r"\s*[.]?\s*$"
            ),
        )

        for pattern in address_patterns:
            match = pattern.search(text)

            if match:
                return match.group(1)

        return None

    # ------------------------------------------------------------------
    # Date of birth
    # ------------------------------------------------------------------

    @classmethod
    def _extract_date_of_birth(
        cls,
        text: str,
    ) -> str | None:
        """
        A date is treated as DOB only when DOB is explicitly labelled.

        Example accepted:

            Date of Birth: 28-07-2008

        Example rejected:

            28-07-2008
        """

        value = cls._find_labeled_value(
            text,
            cls.DATE_OF_BIRTH_TERMS,
        )

        if value:
            return value

        return cls._find_inline_label_value(
            text,
            cls.DATE_OF_BIRTH_TERMS,
        )

    # ------------------------------------------------------------------
    # Explicit person -> field relationships
    # ------------------------------------------------------------------

    @classmethod
    def _extract_person_field_relationship(
        cls,
        text: str,
        person: str,
        field_terms: tuple[str, ...],
    ) -> str | None:
        """
        Extract a value only when the named person and field are
        explicitly connected.

        Accepted forms include:

            John Smith's college: ABC University
            John Smith's college is ABC University
            John Smith's degree: B.E.
            John Smith's phone number: 1234567890

        The method intentionally does NOT accept:

            John Smith
            ABC University

        or:

            John Smith
            ...
            ABC University

        because proximity does not establish a relationship.
        """

        normalized_person = cls._normalize(person)

        if not normalized_person:
            return None

        escaped_person = re.escape(
            normalized_person
        )

        field_pattern = "|".join(
            re.escape(term)
            for term in sorted(
                field_terms,
                key=len,
                reverse=True,
            )
        )

        normalized_text = cls._normalize_whitespace(
            text
        )

        patterns = [
            # John Smith's college: ABC
            re.compile(
                rf"\b{escaped_person}\b"
                rf"\s*['’]s\s+"
                rf"(?:{field_pattern})\b"
                rf"\s*(?::|-|=|is|was|are|were)\s*"
                rf"(?P<value>[^\n\r.;]+)",
                re.IGNORECASE,
            ),

            # John Smith's college is ABC
            re.compile(
                rf"\b{escaped_person}\b"
                rf"\s*['’]s\s+"
                rf"(?:{field_pattern})\b"
                rf"\s+(?:is|was|are|were)\s+"
                rf"(?P<value>[^\n\r.;]+)",
                re.IGNORECASE,
            ),

            # The college of John Smith is ABC
            re.compile(
                rf"\b(?:the\s+)?"
                rf"(?:{field_pattern})\b"
                rf"\s+of\s+"
                rf"{escaped_person}"
                rf"\s*(?:is|was|are|were|:|-|=)\s*"
                rf"(?P<value>[^\n\r.;]+)",
                re.IGNORECASE,
            ),

            # John Smith — College: ABC
            re.compile(
                rf"\b{escaped_person}\b"
                rf"\s*[,|;/\-–—]\s*"
                rf"(?:{field_pattern})\b"
                rf"\s*(?::|=)\s*"
                rf"(?P<value>[^\n\r.;]+)",
                re.IGNORECASE,
            ),

            # John Smith, College: ABC
            re.compile(
                rf"\b{escaped_person}\b"
                rf"\s*,\s*"
                rf"(?:{field_pattern})\b"
                rf"\s*(?::|=)\s*"
                rf"(?P<value>[^\n\r.;]+)",
                re.IGNORECASE,
            ),
        ]

        for pattern in patterns:
            match = pattern.search(
                normalized_text
            )

            if not match:
                continue

            value = match.group(
                "value"
            ).strip()

            value = cls._clean_extracted_value(
                value
            )

            if value:
                return value

        return None

    @classmethod
    def _extract_person_college(
        cls,
        text: str,
        person: str,
    ) -> str | None:
        return cls._extract_person_field_relationship(
            text=text,
            person=person,
            field_terms=cls.COLLEGE_TERMS,
        )

    @classmethod
    def _extract_person_degree(
        cls,
        text: str,
        person: str,
    ) -> str | None:
        return cls._extract_person_field_relationship(
            text=text,
            person=person,
            field_terms=cls.DEGREE_TERMS,
        )

    @classmethod
    def _extract_person_graduation(
        cls,
        text: str,
        person: str,
    ) -> str | None:
        return cls._extract_person_field_relationship(
            text=text,
            person=person,
            field_terms=cls.GRADUATION_TERMS,
        )

    @classmethod
    def _extract_person_identifier(
        cls,
        text: str,
        person: str,
    ) -> str | None:
        return cls._extract_person_field_relationship(
            text=text,
            person=person,
            field_terms=cls.IDENTIFIER_TERMS,
        )

    @classmethod
    def _extract_person_phone(
        cls,
        text: str,
        person: str,
    ) -> str | None:
        return cls._extract_person_field_relationship(
            text=text,
            person=person,
            field_terms=cls.PHONE_TERMS,
        )

    @classmethod
    def _extract_person_email(
        cls,
        text: str,
        person: str,
    ) -> str | None:
        return cls._extract_person_field_relationship(
            text=text,
            person=person,
            field_terms=cls.EMAIL_TERMS,
        )

    @classmethod
    def _extract_person_address(
        cls,
        text: str,
        person: str,
    ) -> str | None:
        return cls._extract_person_field_relationship(
            text=text,
            person=person,
            field_terms=cls.ADDRESS_TERMS,
        )

    # ------------------------------------------------------------------
    # Generic record-aware field extraction
    # ------------------------------------------------------------------

    @classmethod
    def _extract_labeled_field_for_named_person(
        cls,
        text: str,
        person: str,
        labels: tuple[str, ...],
    ) -> str | None:
        """
        Support record-like documents where a named person is explicitly
        identified and the requested field is explicitly labelled.

        Example:

            Candidate Name: John Smith
            College: ABC University

        This is accepted because both the identity and requested field
        are explicit structured fields in the same record.

        This is NOT equivalent to:

            John Smith
            ABC University

        because the latter contains no explicit field relationship.
        """

        normalized_person = cls._normalize(person)

        lines = cls._lines(text)

        person_seen = False

        for line in lines:
            normalized_line = cls._normalize(line)

            # Explicit identity labels.
            identity_match = re.search(
                r"^(?:candidate\s+name|student\s+name|"
                r"applicant\s+name|employee\s+name|"
                r"full\s+name|name)"
                r"\s*(?::|-|=)\s*(.+)$",
                normalized_line,
                re.IGNORECASE,
            )

            if identity_match:
                identity_value = (
                    identity_match.group(1).strip()
                )

                if normalized_person == cls._normalize(
                    identity_value
                ):
                    person_seen = True
                    continue

            # Explicitly named person line.
            if normalized_person == normalized_line:
                person_seen = True
                continue

            if not person_seen:
                continue

            value = cls._find_label_value_in_line(
                line,
                labels,
            )

            if value:
                return value

        return None

    # ------------------------------------------------------------------
    # Examination date
    # ------------------------------------------------------------------

    @classmethod
    def _extract_examination_date(
        cls,
        text: str,
    ) -> str | None:
        value = cls._find_labeled_value(
            text,
            cls.EXAMINATION_DATE_TERMS,
        )

        if value:
            return value

        value = cls._find_inline_label_value(
            text,
            cls.EXAMINATION_DATE_TERMS,
        )

        if value:
            return value

        # Generic date associated with an explicit exam heading.
        exam_context = re.search(
            r"(?is)"
            r"\b(?:exam|examination|test)\b"
            r".{0,120}?"
            r"\b("
            r"(?:\d{1,2}\s+"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
            r"|January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"(?:,\s*|\s+)\d{4})"
            r"|"
            r"(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"\s+\d{1,2},?\s+\d{4}"
            r")\b",
            text,
            re.IGNORECASE,
        )

        if exam_context:
            return exam_context.group(1).strip()

        return None

    # ------------------------------------------------------------------
    # Examination center / venue
    # ------------------------------------------------------------------

    @classmethod
    def _extract_examination_center(
        cls,
        text: str,
    ) -> str | None:
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

        # Generic venue label.
        venue_value = cls._find_labeled_value(
            text,
            (
                "venue",
                "test venue",
                "exam venue",
                "examination venue",
            ),
        )

        if venue_value:
            return venue_value

        return None

    # ------------------------------------------------------------------
    # Registration / identifier
    # ------------------------------------------------------------------

    @classmethod
    def _extract_registration_number(
        cls,
        text: str,
    ) -> str | None:
        value = cls._find_labeled_value(
            text,
            cls.REGISTRATION_NUMBER_TERMS,
        )

        if value:
            return value

        return cls._find_inline_label_value(
            text,
            cls.REGISTRATION_NUMBER_TERMS,
        )

    # ------------------------------------------------------------------
    # Value cleaning
    # ------------------------------------------------------------------

    @classmethod
    def _clean_extracted_value(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        value = value.rstrip(
            " \t,;:|"
        )

        return value

    # ------------------------------------------------------------------
    # Source selection
    # ------------------------------------------------------------------

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

            if normalized_value and normalized_value in text:
                return chunk

        return chunks[0] if chunks else None

    # ------------------------------------------------------------------
    # Structured field detection
    # ------------------------------------------------------------------

    @classmethod
    def _extract_generic_labeled_field(
        cls,
        query: str,
        combined_text: str,
        chunks: list[dict[str, Any]],
        terms: tuple[str, ...],
        field_name: str,
    ) -> dict[str, Any] | None:
        if not cls._query_has_any_term(
            query,
            terms,
        ):
            return None

        value = cls._find_labeled_value(
            combined_text,
            terms,
        )

        if not value:
            value = cls._find_inline_label_value(
                combined_text,
                terms,
            )

        if not value:
            return None

        value = cls._clean_extracted_value(
            value
        )

        if not value:
            return None

        source = cls._find_source_chunk(
            value=value,
            chunks=chunks,
        )

        if source is None:
            return None

        return {
            "value": value,
            "field": field_name,
            "chunk": source,
            "reason": f"explicit_{field_name}_evidence",
        }

    # ------------------------------------------------------------------
    # Person-specific extraction
    # ------------------------------------------------------------------

    @classmethod
    def _extract_person_specific_field(
        cls,
        query: str,
        combined_text: str,
        chunks: list[dict[str, Any]],
        field_terms: tuple[str, ...],
        field_name: str,
        extractor: Any,
    ) -> dict[str, Any] | None:
        person = cls._extract_person_from_query(
            query
        )

        if not person:
            return None

        value = extractor(
            combined_text,
            person,
        )

        if not value:
            value = cls._extract_labeled_field_for_named_person(
                text=combined_text,
                person=person,
                labels=field_terms,
            )

        if not value:
            return None

        value = cls._clean_extracted_value(
            value
        )

        if not value:
            return None

        source = cls._find_source_chunk(
            value=value,
            chunks=chunks,
        )

        if source is None:
            return None

        return {
            "value": value,
            "field": field_name,
            "chunk": source,
            "reason": (
                f"explicit_person_{field_name}_relation"
            ),
        }

    # ------------------------------------------------------------------
    # Main extraction API
    # ------------------------------------------------------------------

    @classmethod
    def extract(
        cls,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Return deterministic evidence only when the requested fact is
        structurally supported.

        Return shape:

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

        # --------------------------------------------------------------
        # Person-specific fields MUST be evaluated first.
        #
        # This prevents a generic "College: ABC" elsewhere in the
        # document from answering "What is John's college?" unless the
        # record explicitly associates the field with John.
        # --------------------------------------------------------------

        person = cls._extract_person_from_query(
            query
        )

        if person:

            person_specific_extractors = [
                (
                    cls.COLLEGE_TERMS,
                    "college",
                    cls._extract_person_college,
                ),
                (
                    cls.DEGREE_TERMS,
                    "degree",
                    cls._extract_person_degree,
                ),
                (
                    cls.GRADUATION_TERMS,
                    "graduation",
                    cls._extract_person_graduation,
                ),
                (
                    cls.IDENTIFIER_TERMS,
                    "identifier",
                    cls._extract_person_identifier,
                ),
                (
                    cls.PHONE_TERMS,
                    "phone",
                    cls._extract_person_phone,
                ),
                (
                    cls.EMAIL_TERMS,
                    "email",
                    cls._extract_person_email,
                ),
                (
                    cls.ADDRESS_TERMS,
                    "address",
                    cls._extract_person_address,
                ),
            ]

            for (
                field_terms,
                field_name,
                extractor,
            ) in person_specific_extractors:
                if not cls._query_has_any_term(
                    query,
                    field_terms,
                ):
                    continue

                result = cls._extract_person_specific_field(
                    query=query,
                    combined_text=combined_text,
                    chunks=chunks,
                    field_terms=field_terms,
                    field_name=field_name,
                    extractor=extractor,
                )

                if result is not None:
                    return result

            # DOB for a named person.
            if cls._query_has_any_term(
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

                return None

        # --------------------------------------------------------------
        # Subject / course
        # --------------------------------------------------------------

        if cls._query_has_any_term(
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

        # --------------------------------------------------------------
        # Postal code
        # --------------------------------------------------------------

        if cls._query_has_any_term(
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

        # --------------------------------------------------------------
        # DOB for document-level question
        # --------------------------------------------------------------

        if cls._query_has_any_term(
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

            # Never treat an arbitrary date as DOB.
            return None

        # --------------------------------------------------------------
        # Examination date
        # --------------------------------------------------------------

        if cls._query_has_any_term(
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

        # --------------------------------------------------------------
        # Examination center
        # --------------------------------------------------------------

        if cls._query_has_any_term(
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

        # --------------------------------------------------------------
        # Registration number
        # --------------------------------------------------------------

        if cls._query_has_any_term(
            query,
            cls.REGISTRATION_NUMBER_TERMS,
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

        # --------------------------------------------------------------
        # Name
        # --------------------------------------------------------------

        if cls._query_has_any_term(
            query,
            cls.NAME_TERMS,
        ):
            value = cls._find_labeled_value(
                combined_text,
                cls.NAME_TERMS,
            )

            if not value:
                value = cls._find_inline_label_value(
                    combined_text,
                    cls.NAME_TERMS,
                )

            if value:
                source = cls._find_source_chunk(
                    value=value,
                    chunks=chunks,
                )

                if source:
                    return {
                        "value": cls._clean_extracted_value(
                            value
                        ),
                        "field": "name",
                        "chunk": source,
                        "reason": (
                            "explicit_name_evidence"
                        ),
                    }

        # --------------------------------------------------------------
        # Generic degree
        # --------------------------------------------------------------

        if cls._query_has_any_term(
            query,
            cls.DEGREE_TERMS,
        ):
            value = cls._find_labeled_value(
                combined_text,
                cls.DEGREE_TERMS,
            )

            if not value:
                value = cls._find_inline_label_value(
                    combined_text,
                    cls.DEGREE_TERMS,
                )

            if value:
                source = cls._find_source_chunk(
                    value=value,
                    chunks=chunks,
                )

                if source:
                    return {
                        "value": cls._clean_extracted_value(
                            value
                        ),
                        "field": "degree",
                        "chunk": source,
                        "reason": (
                            "explicit_degree_evidence"
                        ),
                    }

        # --------------------------------------------------------------
        # Generic graduation
        # --------------------------------------------------------------

        if cls._query_has_any_term(
            query,
            cls.GRADUATION_TERMS,
        ):
            value = cls._find_labeled_value(
                combined_text,
                cls.GRADUATION_TERMS,
            )

            if not value:
                value = cls._find_inline_label_value(
                    combined_text,
                    cls.GRADUATION_TERMS,
                )

            if value:
                source = cls._find_source_chunk(
                    value=value,
                    chunks=chunks,
                )

                if source:
                    return {
                        "value": cls._clean_extracted_value(
                            value
                        ),
                        "field": "graduation",
                        "chunk": source,
                        "reason": (
                            "explicit_graduation_evidence"
                        ),
                    }

        return None


__all__ = [
    "StructuredEvidenceService",
]

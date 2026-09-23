import re
from typing import Any


class ExactEvidenceService:
    """
    Detects exact literal evidence in retrieved document chunks.

    This service does not use outside knowledge and does not infer
    what an identifier means.
    """

    IDENTIFIER_PATTERN = re.compile(
        r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9._:/-]{6,}$"
    )

    MEANING_PATTERNS = (
        re.compile(
            r"\bwhat\s+does\s+(.+?)\s+represent\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhat\s+does\s+(.+?)\s+mean\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhat\s+is\s+the\s+meaning\s+of\s+(.+?)\b",
            re.IGNORECASE,
        ),
    )

    @classmethod
    def _normalize(cls, value: str) -> str:
        return " ".join(
            value.strip().lower().split()
        )

    @classmethod
    def is_literal_identifier(
        cls,
        query: str,
    ) -> bool:
        normalized = query.strip()

        if not normalized:
            return False

        if " " in normalized:
            return False

        return bool(
            cls.IDENTIFIER_PATTERN.fullmatch(
                normalized
            )
        )

    @classmethod
    def find_exact_identifier(
        cls,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Return an exact identifier only when the complete query
        itself is an identifier.
        """

        if not cls.is_literal_identifier(query):
            return None

        normalized_query = cls._normalize(query)

        for chunk in chunks:
            text = str(
                chunk.get("text", "")
            ).strip()

            if not text:
                continue

            normalized_text = cls._normalize(
                text
            )

            if normalized_query in normalized_text:
                return {
                    "value": query.strip(),
                    "chunk": chunk,
                }

        return None

    @classmethod
    def find_identifier_in_meaning_question(
        cls,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Detect questions asking what an explicitly present identifier
        represents.

        This does not determine the identifier's meaning.

        It only establishes:
        1. the question is asking for meaning, and
        2. an identifier occurs in the retrieved evidence.
        """

        identifier = None

        # Search the original query so the returned identifier preserves
        # the exact casing supplied by the user.
        for pattern in cls.MEANING_PATTERNS:
            match = pattern.search(query.strip())

            if match:
                candidate = match.group(1).strip()

                if cls.is_literal_identifier(candidate):
                    identifier = candidate
                    break

        if identifier is None:
            return None

        normalized_identifier = cls._normalize(
            identifier
        )

        for chunk in chunks:
            text = str(
                chunk.get("text", "")
            ).strip()

            if not text:
                continue

            normalized_text = cls._normalize(
                text
            )

            if normalized_identifier in normalized_text:
                return {
                    "identifier": identifier,
                    "chunk": chunk,
                }

        return None

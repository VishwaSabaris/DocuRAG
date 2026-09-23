from difflib import SequenceMatcher
from typing import Any

from app.core.config import get_settings
from app.services.exact_evidence.exact_evidence_service import (
    ExactEvidenceService,
)
from app.services.llm.ollama_service import OllamaService
from app.services.structured_evidence.structured_evidence_service import (
    StructuredEvidenceService,
)


DEFAULT_SYSTEM_PROMPT = """
You are DocuRAG, a strict document question-answering assistant.

Your ONLY source of truth is the document context provided in the user
prompt.

GROUNDING RULES:

1. Use ONLY facts explicitly stated in the document context.
2. NEVER use outside knowledge or your own assumptions.
3. NEVER infer a person's role, job, relationship, identity, ownership,
   affiliation, or responsibility unless the document explicitly states it.
4. NEVER infer relationships between fields merely because they appear next
   to each other.
5. Do NOT infer relationships merely from physical or textual proximity.
6. A name appearing in a document does NOT tell you the person's role.
7. A number appearing next to a name does NOT establish what that number
   represents unless the document explicitly explains it.
8. A date appearing in a document does NOT automatically mean it is a
   date of birth.
9. A number appearing in an address may be a postal code only when the
   address structure explicitly supports that interpretation.
10. An institution name appearing in a document does NOT establish that it
    belongs to a named person unless the relationship is explicitly stated.
11. Do NOT infer relationships between fields merely because they appear
    next to each other.
12. If the document contains a name but does not explain who the person is,
    say that the document lists the name but does not specify the person's
    role or identity.
13. If the requested information is not explicitly supported by the
    document context, say:
    "The information is not available in the provided document."
14. Keep the answer concise and directly answer the question.
15. When answering from a source, mention the page number when available.

LITERAL VALUE PRESERVATION:

16. When the answer is a literal value explicitly present in the document,
    preserve the COMPLETE value exactly as written in the source.

17. Never shorten, paraphrase, normalize, or silently remove any part of
    an explicitly stated name, identifier, code, date, number, address,
    organization name, or other literal value.

18. Preserve:
    - initials
    - suffixes
    - prefixes
    - punctuation
    - digits
    - hyphens
    - capitalization when practical
    - complete multi-word names
    - complete identifiers and codes

19. If the answer is a location or organization name, preserve the complete
    explicitly stated name rather than removing parts that appear redundant.

20. Do not combine separate nearby fields into one answer unless the
    document explicitly establishes that they belong together.

21. If there is uncertainty about which literal value answers the question,
    do not guess.

IMPORTANT:

The document context may contain multiple unrelated fields.

Do NOT infer relationships between fields merely because they appear
next to each other.

Do NOT infer relationships from physical or textual proximity.

If the context says:

"John Smith
123456
Computer Science"

you MUST NOT conclude that John Smith is a professor, student,
employee, instructor, or owner unless the document explicitly says so.

Likewise, the presence of:

"John Smith
ABC Engineering College"

does NOT establish that ABC Engineering College is John's college
unless the document explicitly establishes that relationship.

Likewise, the presence of:

"John Smith
28-07-2008"

does NOT establish that 28-07-2008 is John Smith's date of birth
unless the document explicitly labels or establishes it as such.

If the requested information is completely absent, say:

"The information is not available in the provided document."
""".strip()


DEFAULT_NOT_AVAILABLE_MESSAGE = (
    "The information is not available in the provided document."
)


class GenerationService:
    """
    Grounded document answer generation.

    Pipeline:

        Retrieved chunks
              ↓
        Deduplication
              ↓
        Context budget
              ↓
        Structured evidence
              ↓
        Exact identifier evidence
              ↓
        Meaning-question evidence
              ↓
        Grounded LLM
    """

    NEAR_DUPLICATE_THRESHOLD = 0.90

    def __init__(
        self,
        ollama_service: OllamaService | None = None,
        max_context_characters: int | None = None,
    ) -> None:
        self.ollama_service = (
            ollama_service
            if ollama_service is not None
            else OllamaService()
        )

        settings = get_settings()

        self.max_context_characters = (
            max_context_characters
            if max_context_characters is not None
            else settings.max_context_characters
        )

        if self.max_context_characters <= 0:
            raise ValueError(
                "max_context_characters must be greater than 0."
            )

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        return " ".join(
            text.strip().split()
        ).lower()

    @classmethod
    def _is_duplicate_or_redundant(
        cls,
        candidate_text: str,
        selected_text: str,
    ) -> bool:
        candidate = cls._normalize_text(
            candidate_text
        )

        selected = cls._normalize_text(
            selected_text
        )

        if not candidate or not selected:
            return False

        if candidate == selected:
            return True

        if (
            candidate in selected
            or selected in candidate
        ):
            return True

        similarity = SequenceMatcher(
            None,
            candidate,
            selected,
        ).ratio()

        return (
            similarity
            >= cls.NEAR_DUPLICATE_THRESHOLD
        )

    def deduplicate_chunks(
        self,
        retrieved_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not retrieved_chunks:
            return []

        selected_chunks: list[dict[str, Any]] = []
        selected_texts: list[str] = []
        selected_chunk_ids: set[str] = set()

        for chunk in retrieved_chunks:
            chunk_id = chunk.get(
                "chunk_id"
            )

            chunk_id_key = (
                str(chunk_id)
                if chunk_id is not None
                else ""
            )

            if (
                chunk_id_key
                and chunk_id_key in selected_chunk_ids
            ):
                continue

            text = str(
                chunk.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            is_redundant = any(
                self._is_duplicate_or_redundant(
                    candidate_text=text,
                    selected_text=selected_text,
                )
                for selected_text in selected_texts
            )

            if is_redundant:
                continue

            selected_chunk = dict(
                chunk
            )

            selected_chunks.append(
                selected_chunk
            )

            selected_texts.append(
                text
            )

            if chunk_id_key:
                selected_chunk_ids.add(
                    chunk_id_key
                )

        return selected_chunks

    def apply_context_budget(
        self,
        retrieved_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not retrieved_chunks:
            return []

        selected_chunks: list[dict[str, Any]] = []
        total_characters = 0

        for chunk in retrieved_chunks:
            text = str(
                chunk.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            chunk_characters = len(
                text
            )

            if not selected_chunks:
                selected_chunks.append(
                    dict(chunk)
                )

                total_characters = (
                    chunk_characters
                )

                continue

            if (
                total_characters
                + chunk_characters
                > self.max_context_characters
            ):
                break

            selected_chunks.append(
                dict(chunk)
            )

            total_characters += (
                chunk_characters
            )

        return selected_chunks

    def build_context(
        self,
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        if not retrieved_chunks:
            return ""

        context_parts: list[str] = []

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            page_number = chunk.get(
                "page_number",
                "unknown",
            )

            text = str(
                chunk.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            context_parts.append(
                (
                    f"[Source {index} | "
                    f"Page {page_number}]\n"
                    f"{text}"
                )
            )

        return "\n\n".join(
            context_parts
        )

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        context = self.build_context(
            retrieved_chunks
        )

        if not context:
            context = (
                "No relevant document context "
                "was retrieved."
            )

        return f"""
DOCUMENT CONTEXT
================

{context}

QUESTION
========

{query}

INSTRUCTIONS
============

Answer the question using ONLY information explicitly stated
in the DOCUMENT CONTEXT.

Before answering, check whether the requested fact is explicitly
supported by the context.

If the question asks you to find, repeat, extract, or identify a
literal value such as an identifier, registration number, code, name,
date, address, organization name, or other exact text, and that value
appears explicitly in the DOCUMENT CONTEXT, return the COMPLETE value
exactly as written in the source.

Do not shorten a literal value.

Do NOT infer relationships.

Do NOT infer relationships between fields merely because they appear
next to each other.

Do NOT infer a relationship merely from physical or textual proximity.

For example:

John Smith
123456
Computer Science

does NOT establish that 123456 is John Smith's registration number
unless the document explicitly says so.

Likewise:

John Smith
28-07-2008

does NOT establish that 28-07-2008 is John Smith's date of birth
unless the document explicitly labels or establishes it as such.

Likewise:

John Smith
ABC Engineering College

does NOT establish that ABC Engineering College is John's college
unless the document explicitly establishes that relationship.

For an address such as:

Coimbatore, Tamil Nadu, India, 641109

the final six-digit value may be returned as a postal code when the
document structure explicitly presents it as part of the address.

If the requested information is completely absent from the provided
document context, say:

"The information is not available in the provided document."

If the requested information is not explicitly supported by the
document context, say:

"The information is not available in the provided document."

Keep the answer concise.

ANSWER
======
""".strip()

    @staticmethod
    def _build_sources(
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "score": chunk["score"],
                "rerank_score": chunk.get(
                    "rerank_score"
                ),
            }
            for chunk in chunks
        ]

    @staticmethod
    def _build_meaning_unavailable_answer(
        identifier: str,
    ) -> str:
        return (
            f"The document lists the code {identifier} "
            "but does not specify what it represents."
        )

    @staticmethod
    def _build_structured_evidence_answer(
        value: str,
    ) -> str:
        return value

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        # ---------------------------------------------------------
        # 1. Deduplicate
        # ---------------------------------------------------------

        context_chunks = self.deduplicate_chunks(
            retrieved_chunks
        )

        # ---------------------------------------------------------
        # 2. Context budget
        # ---------------------------------------------------------

        context_chunks = self.apply_context_budget(
            context_chunks
        )

        if not context_chunks:
            return {
                "answer": DEFAULT_NOT_AVAILABLE_MESSAGE,
                "sources": [],
            }

        # ---------------------------------------------------------
        # 3. Structured evidence
        # ---------------------------------------------------------

        structured_evidence = (
            StructuredEvidenceService.extract(
                query=query,
                chunks=context_chunks,
            )
        )

        if structured_evidence is not None:
            return {
                "answer": self._build_structured_evidence_answer(
                    structured_evidence["value"]
                ),
                "sources": self._build_sources(
                    [
                        structured_evidence[
                            "chunk"
                        ]
                    ]
                ),
            }

        # ---------------------------------------------------------
        # 4. Meaning-question evidence
        # ---------------------------------------------------------

        meaning_evidence = (
            ExactEvidenceService
            .find_identifier_in_meaning_question(
                query=query,
                chunks=context_chunks,
            )
        )

        if meaning_evidence is not None:
            identifier = (
                meaning_evidence[
                    "identifier"
                ]
            )

            return {
                "answer": (
                    self._build_meaning_unavailable_answer(
                        identifier
                    )
                ),
                "sources": self._build_sources(
                    [
                        meaning_evidence[
                            "chunk"
                        ]
                    ]
                ),
            }

        # ---------------------------------------------------------
        # 5. Exact identifier
        # ---------------------------------------------------------

        exact_evidence = (
            ExactEvidenceService.find_exact_identifier(
                query=query,
                chunks=context_chunks,
            )
        )

        if exact_evidence is not None:
            source_chunk = (
                exact_evidence["chunk"]
            )

            return {
                "answer": exact_evidence["value"],
                "sources": self._build_sources(
                    [source_chunk]
                ),
            }

        # ---------------------------------------------------------
        # 6. Grounded LLM generation
        # ---------------------------------------------------------

        prompt = self.build_prompt(
            query=query,
            retrieved_chunks=context_chunks,
        )

        answer = self.ollama_service.generate(
            prompt=prompt,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )

        sources = self._build_sources(
            context_chunks
        )

        return {
            "answer": answer,
            "sources": sources,
        }


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_NOT_AVAILABLE_MESSAGE",
    "GenerationService",
]

from typing import Any
from uuid import UUID

from psycopg import Connection

from app.core.config import get_settings
from app.repositories.document_repository import DocumentRepository
from app.services.embeddings.embedding_service import (
    EmbeddingService,
    get_embedding_service,
)

from typing import Any
from uuid import UUID

from psycopg import Connection

from app.core.config import get_settings
from app.repositories.document_repository import DocumentRepository
from app.services.embeddings.embedding_service import (
    EmbeddingService,
    get_embedding_service,
)


class HybridRetrievalService:
    """
    Hybrid retrieval service combining:

        Vector Search
              +
        PostgreSQL Full-Text Search
              +
        Exact Keyword Search
              ↓
        Reciprocal Rank Fusion (RRF)
              ↓
        Exact-Match Boosting
              ↓
        Candidate Ranking

    The resulting candidates can then be passed to the
    cross-encoder reranker.

    Vector search is useful for semantic similarity.

    Full-text search is useful for lexical matching using
    PostgreSQL's indexed tsvector representation.

    Exact keyword search is useful for literal substring
    matches such as:
        - names
        - IDs
        - registration numbers
        - dates
        - addresses
        - exact phrases

    Exact-match boosting gives an additional deterministic
    ranking signal when the complete query appears inside
    the candidate chunk.

    Adaptive semantic recovery is used when normal hybrid
    retrieval produces no candidates. This allows legitimate
    low-similarity semantic questions to reach the reranker
    without globally lowering the normal semantic threshold.
    """

    RRF_K = 60

    # Additional score applied when the complete query appears
    # as a case-insensitive substring of the candidate text.
    EXACT_MATCH_BOOST = 0.10

    # Recovery floor used only when normal hybrid retrieval
    # produces no candidates.
    #
    # This is intentionally lower than the normal retrieval
    # threshold but still excludes very weak semantic matches.
    RECOVERY_MIN_SCORE = 0.08

    def __init__(
        self,
        connection: Connection,
        embedding_service: EmbeddingService | None = None,
        min_score: float | None = None,
    ) -> None:
        self.repository = DocumentRepository(connection)

        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else get_embedding_service()
        )

        settings = get_settings()

        self.min_score = (
            min_score
            if min_score is not None
            else settings.retrieval_min_score
        )

        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(
                "min_score must be between 0.0 and 1.0."
            )

        if not 0.0 <= self.RECOVERY_MIN_SCORE <= 1.0:
            raise ValueError(
                "RECOVERY_MIN_SCORE must be between 0.0 and 1.0."
            )

    @classmethod
    def _rrf_score(cls, rank: int) -> float:
        """
        Calculate Reciprocal Rank Fusion contribution.

        RRF formula:

            1 / (k + rank)

        where k = 60.
        """

        if rank <= 0:
            raise ValueError(
                "RRF rank must be greater than 0."
            )

        return 1.0 / (cls.RRF_K + rank)

    @classmethod
    def _has_exact_match(
        cls,
        query: str,
        text: str,
    ) -> bool:
        """
        Determine whether the complete query occurs inside
        the candidate text.

        Matching is case-insensitive and whitespace-normalized.
        """

        normalized_query = " ".join(
            query.strip().lower().split()
        )

        normalized_text = " ".join(
            text.strip().lower().split()
        )

        if not normalized_query:
            return False

        return normalized_query in normalized_text

    def retrieve(
        self,
        query: str,
        vector_top_k: int = 10,
        keyword_top_k: int = 10,
        document_id: UUID | None = None,
        full_text_top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve candidate document chunks using hybrid retrieval.

        Retrieval stages:
            1. Vector similarity search
            2. Exact keyword search
            3. PostgreSQL full-text search
            4. Reciprocal Rank Fusion
            5. Exact-match boosting
            6. Adaptive semantic recovery when necessary
            7. Final candidate ordering

        The returned candidates are intended to be passed to
        the cross-encoder reranker.

        Normal semantic retrieval uses the configured
        minimum similarity threshold.

        If normal hybrid retrieval produces no candidates,
        low-confidence vector results are reconsidered using
        RECOVERY_MIN_SCORE. This improves recall for questions
        whose wording is semantically related to the document
        but does not share literal terminology with it.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if vector_top_k <= 0:
            raise ValueError(
                "vector_top_k must be greater than 0."
            )

        if keyword_top_k <= 0:
            raise ValueError(
                "keyword_top_k must be greater than 0."
            )

        if full_text_top_k is None:
            full_text_top_k = keyword_top_k

        if full_text_top_k <= 0:
            raise ValueError(
                "full_text_top_k must be greater than 0."
            )

        query_embedding = self.embedding_service.embed_text(
            query
        )

        vector_results = self.repository.similarity_search(
            query_embedding=query_embedding,
            top_k=vector_top_k,
            document_id=document_id,
        )

        keyword_results = self.repository.keyword_search(
            query_text=query,
            top_k=keyword_top_k,
            document_id=document_id,
        )

        full_text_results = self.repository.full_text_search(
            query_text=query,
            top_k=full_text_top_k,
            document_id=document_id,
        )

        candidates: dict[str, dict[str, Any]] = {}

        def get_or_create_candidate(
            result: dict[str, Any],
        ) -> dict[str, Any]:
            chunk_id = str(result["id"])

            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "chunk_id": result["id"],
                    "document_id": result["document_id"],
                    "page_number": result["page_number"],
                    "chunk_index": result["chunk_index"],
                    "text": result["text"],
                    "character_count": result["character_count"],
                    "score": 0.0,
                    "distance": None,
                    "keyword_score": 0.0,
                    "lexical_score": 0.0,
                    "rrf_score": 0.0,
                    "exact_match_score": 0.0,
                    "exact_match_boost": 0.0,
                    "hybrid_score": 0.0,
                    "retrieval_methods": [],
                    "retrieval_ranks": {},
                }

            return candidates[chunk_id]

        # --------------------------------------------------
        # Normal vector retrieval
        # --------------------------------------------------

        for rank, result in enumerate(
            vector_results,
            start=1,
        ):
            distance = float(result["distance"])
            similarity_score = 1.0 - distance

            # Preserve the normal semantic similarity
            # filtering behavior.
            if similarity_score < self.min_score:
                continue

            candidate = get_or_create_candidate(result)

            candidate["score"] = similarity_score
            candidate["distance"] = distance

            candidate["rrf_score"] += self._rrf_score(
                rank
            )

            candidate["retrieval_methods"].append(
                "vector"
            )

            candidate["retrieval_ranks"]["vector"] = rank

        # --------------------------------------------------
        # Exact keyword retrieval
        # --------------------------------------------------

        for rank, result in enumerate(
            keyword_results,
            start=1,
        ):
            candidate = get_or_create_candidate(result)

            keyword_score = float(
                result["keyword_score"]
            )

            candidate["keyword_score"] = keyword_score

            candidate["rrf_score"] += self._rrf_score(
                rank
            )

            candidate["retrieval_methods"].append(
                "keyword"
            )

            candidate["retrieval_ranks"]["keyword"] = rank

        # --------------------------------------------------
        # PostgreSQL full-text retrieval
        # --------------------------------------------------

        for rank, result in enumerate(
            full_text_results,
            start=1,
        ):
            candidate = get_or_create_candidate(result)

            lexical_score = float(
                result["lexical_score"]
            )

            candidate["lexical_score"] = lexical_score

            candidate["rrf_score"] += self._rrf_score(
                rank
            )

            candidate["retrieval_methods"].append(
                "full_text"
            )

            candidate["retrieval_ranks"][
                "full_text"
            ] = rank

        # --------------------------------------------------
        # Adaptive semantic recovery
        # --------------------------------------------------
        #
        # If normal hybrid retrieval found nothing, reconsider
        # the vector results using a lower recovery floor.
        #
        # This specifically handles queries such as:
        #
        #     "Who is the candidate?"
        #
        # where the document contains:
        #
        #     "Vishwa Sabaris V"
        #
        # but does not literally contain the word "candidate".
        #
        # Very weak unrelated queries remain excluded.
        # --------------------------------------------------

        if not candidates:
            for rank, result in enumerate(
                vector_results,
                start=1,
            ):
                distance = float(result["distance"])
                similarity_score = 1.0 - distance

                if (
                    similarity_score
                    < self.RECOVERY_MIN_SCORE
                ):
                    continue

                candidate = get_or_create_candidate(
                    result
                )

                candidate["score"] = similarity_score
                candidate["distance"] = distance

                candidate["rrf_score"] += (
                    self._rrf_score(rank)
                )

                candidate["retrieval_methods"].append(
                    "vector_recovery"
                )

                candidate["retrieval_ranks"][
                    "vector_recovery"
                ] = rank

        # --------------------------------------------------
        # Remove duplicate retrieval method names
        # --------------------------------------------------

        for candidate in candidates.values():
            candidate["retrieval_methods"] = list(
                dict.fromkeys(
                    candidate["retrieval_methods"]
                )
            )

        # --------------------------------------------------
        # Exact-match boosting
        # --------------------------------------------------

        for candidate in candidates.values():
            exact_match = self._has_exact_match(
                query=query,
                text=str(candidate["text"]),
            )

            if exact_match:
                candidate["exact_match_score"] = 1.0
                candidate["exact_match_boost"] = (
                    self.EXACT_MATCH_BOOST
                )
            else:
                candidate["exact_match_score"] = 0.0
                candidate["exact_match_boost"] = 0.0

            candidate["hybrid_score"] = (
                candidate["rrf_score"]
                + candidate["exact_match_boost"]
            )

        # --------------------------------------------------
        # Final hybrid candidate ordering
        # --------------------------------------------------

        return sorted(
            candidates.values(),
            key=lambda item: (
                -item["hybrid_score"],
                -item["rrf_score"],
                -item["lexical_score"],
                -item["keyword_score"],
                item["chunk_index"],
            ),
        )

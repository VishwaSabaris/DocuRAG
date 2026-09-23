from functools import lru_cache
from typing import Any


DEFAULT_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


class RerankingService:
    """
    Local cross-encoder reranking service.

    The reranker receives the user's query together with
    candidate document chunks and assigns a relevance score
    to each query-document pair.

    Pipeline:

        query + candidate chunks
                    ↓
              cross-encoder
                    ↓
             reranking scores
                    ↓
              sorted chunks
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Rerank retrieved document chunks.

        Args:
            query:
                User's natural-language question.

            documents:
                Candidate chunks returned by vector retrieval.

            top_k:
                Maximum number of chunks to return.

        Returns:
            The highest-ranked chunks with an additional
            rerank_score field.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if not documents:
            return []

        pairs = [
            (
                query,
                str(document.get("text", "")),
            )
            for document in documents
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked_documents: list[dict[str, Any]] = []

        for document, score in zip(
            documents,
            scores,
        ):
            reranked_document = dict(document)

            reranked_document[
                "rerank_score"
            ] = float(score)

            reranked_documents.append(
                reranked_document
            )

        reranked_documents.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked_documents[:top_k]


@lru_cache(maxsize=1)
def get_reranking_service() -> RerankingService:
    """
    Return a cached reranking service.

    The cross-encoder model is loaded only once per
    application process.
    """

    return RerankingService()

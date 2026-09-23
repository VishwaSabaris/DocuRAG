from functools import lru_cache

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


class EmbeddingService:
    """
    Local embedding service using Sentence Transformers.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        if dimension is None:
            raise RuntimeError(
                "Unable to determine embedding dimension."
            )

        return int(dimension)

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        if not text.strip():
            raise ValueError(
                "Cannot embed empty text."
            )

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        if any(
            not text.strip()
            for text in texts
        ):
            raise ValueError(
                "All texts must contain "
                "non-whitespace characters."
            )

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

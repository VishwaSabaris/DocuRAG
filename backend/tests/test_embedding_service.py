from app.services.embeddings.embedding_service import EmbeddingService


def test_embedding_dimension() -> None:
    service = EmbeddingService()

    assert service.dimension == 384


def test_single_embedding() -> None:
    service = EmbeddingService()

    embedding = service.embed_text(
        "The application collects job listings."
    )

    assert len(embedding) == 384
    assert all(isinstance(value, float) for value in embedding)


def test_multiple_embeddings() -> None:
    service = EmbeddingService()

    texts = [
        "The application collects job listings.",
        "The system validates scraped information.",
    ]

    embeddings = service.embed_documents(texts)

    assert len(embeddings) == 2
    assert all(len(embedding) == 384 for embedding in embeddings)

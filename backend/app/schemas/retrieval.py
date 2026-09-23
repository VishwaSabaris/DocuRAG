from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """
    Request body for semantic document retrieval.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language question or search query.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of chunks to retrieve.",
    )


class RetrievalResult(BaseModel):
    """
    A single retrieved document chunk.
    """

    chunk_id: UUID
    document_id: UUID
    page_number: int
    chunk_index: int
    text: str
    character_count: int
    score: float
    distance: float


class RetrievalResponse(BaseModel):
    """
    Response returned by the semantic retrieval endpoint.
    """

    document_id: UUID
    query: str
    results: list[RetrievalResult]


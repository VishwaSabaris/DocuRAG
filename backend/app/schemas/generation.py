from uuid import UUID

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """
    Request body for document question answering.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language question about the document.",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of final chunks used for generation.",
    )


class GenerationSource(BaseModel):
    """
    Source information used to generate the answer.
    """

    chunk_id: UUID
    document_id: UUID
    page_number: int
    chunk_index: int
    score: float
    rerank_score: float | None = None


class GenerationResponse(BaseModel):
    """
    Response returned by the document question-answering endpoint.
    """

    document_id: UUID
    query: str
    answer: str
    sources: list[GenerationSource]

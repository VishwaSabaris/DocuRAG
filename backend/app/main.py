from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.db.database import pool
from app.repositories.document_repository import DocumentRepository
from app.schemas.generation import (
    GenerationRequest,
    GenerationResponse,
    GenerationSource,
)
from app.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
)
from app.services.answerability.answerability_service import (
    AnswerabilityService,
)
from app.services.generation.generation_service import (
    GenerationService,
)
from app.services.ingestion.document_ingestion_service import (
    DocumentIngestionService,
)
from app.services.ingestion.pdf_service import PDFProcessingError
from app.services.reranking.reranking_service import (
    get_reranking_service,
)
from app.services.retrieval.hybrid_retrieval_service import (
    HybridRetrievalService,
)


settings = get_settings()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-oriented local RAG platform "
        "using Gemma and Ollama."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parents[2]

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


NOT_AVAILABLE_MESSAGE = (
    "The information is not available "
    "in the provided document."
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "Welcome to DocuRAG",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
) -> dict:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    safe_filename = Path(file.filename).name

    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    document_id = uuid4()

    stored_filename = (
        f"{document_id}_{safe_filename}"
    )

    file_path = UPLOAD_DIR / stored_filename

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        file_path.write_bytes(contents)

        with pool.connection() as connection:
            ingestion_service = (
                DocumentIngestionService(
                    connection=connection,
                )
            )

            result = ingestion_service.ingest_pdf(
                file_path=file_path,
                document_id=document_id,
            )

        document = result["document"]

        return {
            "document_id": str(document["id"]),
            "filename": document["filename"],
            "stored_filename": document[
                "stored_filename"
            ],
            "status": document["status"],
            "page_count": result["page_count"],
            "total_character_count": result[
                "total_character_count"
            ],
            "chunk_count": result[
                "chunk_count"
            ],
            "embedding_count": result[
                "embedding_count"
            ],
        }

    except PDFProcessingError as exc:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except HTTPException:
        if file_path.exists():
            file_path.unlink()

        raise

    except Exception as exc:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}",
        ) from exc


# ============================================================
# DOCUMENT SEARCH
# ============================================================

@app.post(
    "/documents/{document_id}/search",
    response_model=RetrievalResponse,
)
async def search_document(
    document_id: UUID,
    request: RetrievalRequest,
) -> RetrievalResponse:

    with pool.connection() as connection:
        repository = DocumentRepository(
            connection
        )

        document = repository.get_document(
            document_id=document_id
        )

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        if document["status"] != "processed":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Document is not ready for search. "
                    f"Current status: "
                    f"{document['status']}"
                ),
            )

        # ----------------------------------------------------
        # 1. Hybrid retrieval
        # ----------------------------------------------------

        retrieval_service = (
            HybridRetrievalService(
                connection=connection
            )
        )

        try:
            results = retrieval_service.retrieve(
                query=request.query,
                vector_top_k=20,
                keyword_top_k=20,
                full_text_top_k=20,
                document_id=document_id,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        if not results:
            return RetrievalResponse(
                document_id=document_id,
                query=request.query,
                results=[],
            )

        # ----------------------------------------------------
        # 2. Cross-encoder reranking
        # ----------------------------------------------------

        reranking_service = (
            get_reranking_service()
        )

        try:
            reranked_results = (
                reranking_service.rerank(
                    query=request.query,
                    documents=results,
                    top_k=request.top_k,
                )
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        return RetrievalResponse(
            document_id=document_id,
            query=request.query,
            results=[
                RetrievalResult(**result)
                for result in reranked_results
            ],
        )


# ============================================================
# DOCUMENT QUESTION ANSWERING
# ============================================================

@app.post(
    "/documents/{document_id}/ask",
    response_model=GenerationResponse,
)
async def ask_document(
    document_id: UUID,
    request: GenerationRequest,
) -> GenerationResponse:

    with pool.connection() as connection:
        repository = DocumentRepository(
            connection
        )

        # ----------------------------------------------------
        # 0. Validate document
        # ----------------------------------------------------

        document = repository.get_document(
            document_id=document_id
        )

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        if document["status"] != "processed":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Document is not ready for "
                    "questions. "
                    f"Current status: "
                    f"{document['status']}"
                ),
            )

        # ----------------------------------------------------
        # 1. Hybrid retrieval
        # ----------------------------------------------------

        retrieval_service = (
            HybridRetrievalService(
                connection=connection
            )
        )

        try:
            retrieved_chunks = (
                retrieval_service.retrieve(
                    query=request.query,
                    vector_top_k=20,
                    keyword_top_k=20,
                    full_text_top_k=20,
                    document_id=document_id,
                )
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        if not retrieved_chunks:
            return GenerationResponse(
                document_id=document_id,
                query=request.query,
                answer=NOT_AVAILABLE_MESSAGE,
                sources=[],
            )

        # ----------------------------------------------------
        # 2. Cross-encoder reranking
        # ----------------------------------------------------

        reranking_service = (
            get_reranking_service()
        )

        try:
            reranked_chunks = (
                reranking_service.rerank(
                    query=request.query,
                    documents=retrieved_chunks,
                    top_k=request.top_k,
                )
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        if not reranked_chunks:
            return GenerationResponse(
                document_id=document_id,
                query=request.query,
                answer=NOT_AVAILABLE_MESSAGE,
                sources=[],
            )

        # ----------------------------------------------------
        # 3. Answerability gate
        # ----------------------------------------------------

        answerability_service = (
            AnswerabilityService()
        )

        try:
            answerability = (
                answerability_service.check(
                    query=request.query,
                    reranked_chunks=reranked_chunks,
                )
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        if not answerability["answerable"]:
            return GenerationResponse(
                document_id=document_id,
                query=request.query,
                answer=NOT_AVAILABLE_MESSAGE,
                sources=[],
            )

        # ----------------------------------------------------
        # 4. Grounded generation
        # ----------------------------------------------------

        generation_service = GenerationService()

        try:
            result = (
                generation_service.generate_answer(
                    query=request.query,
                    retrieved_chunks=reranked_chunks,
                )
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except RuntimeError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"LLM generation failed: {exc}"
                ),
            ) from exc

        # ----------------------------------------------------
        # 5. Build final sources
        #
        # IMPORTANT:
        #
        # GenerationService._build_sources()
        # may already include rerank_score.
        #
        # Therefore we must NOT do:
        #
        # GenerationSource(
        #     **source,
        #     rerank_score=...
        # )
        #
        # because that causes:
        #
        # TypeError:
        # got multiple values for keyword argument
        # 'rerank_score'
        #
        # Instead, explicitly construct the Pydantic model
        # and safely obtain rerank_score from the source itself.
        # ----------------------------------------------------

        generation_sources = []

        for source in result.get(
            "sources",
            [],
        ):
            generation_sources.append(
                GenerationSource(
                    chunk_id=source[
                        "chunk_id"
                    ],
                    document_id=source[
                        "document_id"
                    ],
                    page_number=source[
                        "page_number"
                    ],
                    chunk_index=source[
                        "chunk_index"
                    ],
                    score=source[
                        "score"
                    ],
                    rerank_score=source.get(
                        "rerank_score"
                    ),
                )
            )

        # ----------------------------------------------------
        # 6. Final API response
        # ----------------------------------------------------

        return GenerationResponse(
            document_id=document_id,
            query=request.query,
            answer=result["answer"],
            sources=generation_sources,
        )

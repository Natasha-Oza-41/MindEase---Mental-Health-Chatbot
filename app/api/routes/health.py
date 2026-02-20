from fastapi import APIRouter, Depends

from app.core.memory import ConversationMemory, get_memory
from app.core.rag_pipeline import RAGPipeline, get_rag_pipeline
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    description="Returns the status of the API, RAG pipeline, and active sessions.",
)
async def health_check(
    rag: RAGPipeline = Depends(get_rag_pipeline),
    memory: ConversationMemory = Depends(get_memory),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="1.0.0",
        rag_status="ready" if rag.is_ready() else "not_initialized — run ingest script",
        active_sessions=memory.session_count(),
    )

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService, get_chat_service
from app.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the chatbot",
    description=(
        "Submit a user message and receive an AI-generated mental health support "
        "response. The session_id keeps conversation history isolated per user."
    ),
)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    try:
        return service.process(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error(f"Runtime error in chat endpoint: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.delete(
    "/chat/{session_id}",
    summary="Clear a conversation session",
    description="Wipe all conversation history for the given session_id.",
)
async def clear_session(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
) -> dict:
    service.memory.clear_session(session_id)
    return {"message": f"Session '{session_id}' has been cleared."}

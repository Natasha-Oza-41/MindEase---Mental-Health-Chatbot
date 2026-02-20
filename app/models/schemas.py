from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class CrisisSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CrisisResource(BaseModel):
    name: str
    contact: str
    available: str
    url: str


class CrisisInfo(BaseModel):
    severity: CrisisSeverity
    detected_keywords: List[str]
    resources: List[CrisisResource]


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique identifier for the user session")
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's message",
    )


class ChatResponse(BaseModel):
    session_id: str
    response: str
    crisis_info: Optional[CrisisInfo] = None
    sources: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    rag_status: str
    active_sessions: int

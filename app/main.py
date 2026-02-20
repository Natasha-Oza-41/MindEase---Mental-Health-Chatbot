"""
Application entry point.

The lifespan context manager handles startup and shutdown logic cleanly
(preferred over the deprecated @app.on_event pattern in FastAPI 0.93+).
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, health
from app.config import get_settings
from app.core.rag_pipeline import get_rag_pipeline
from app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger("main", settings.log_level)

STATIC_DIR = Path("static")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # ── Startup ────────────────────────────────────────────────────────
    logger.info("Starting Mental Health Chatbot API...")
    rag = get_rag_pipeline()
    if rag.is_ready():
        logger.info("RAG pipeline is ready.")
    else:
        logger.warning(
            "RAG pipeline not initialized. "
            "Run:  python scripts/ingest_documents.py"
        )

    yield

    # ── Shutdown ───────────────────────────────────────────────────────
    logger.info("Shutting down - goodbye.")


app = FastAPI(
    title="Mental Health Chatbot API",
    description=(
        "An AI-powered mental health support chatbot with RAG, "
        "session memory, and a crisis detection safety layer."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])

# Serve frontend static assets (/static/style.css, /static/app.js, etc.)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", tags=["Frontend"], include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serve the chat frontend."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    # Fallback JSON if frontend not built yet
    from fastapi.responses import JSONResponse
    return JSONResponse({
        "service": "Mental Health Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "frontend": "static/index.html not found",
    })

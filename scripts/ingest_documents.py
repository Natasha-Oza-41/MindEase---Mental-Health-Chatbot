#!/usr/bin/env python3
"""
Document ingestion script.

Run this once before starting the server (and re-run whenever you add
or update documents in data/mental_health_docs/).

Usage:
    python scripts/ingest_documents.py
"""
import sys
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.core.rag_pipeline import get_rag_pipeline
from app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger("ingest")


def main() -> None:
    print("\n Mental Health Chatbot - Document Ingestion")
    print("=" * 50)
    print(f"Documents path : {settings.documents_path}")
    print(f"Vector store   : {settings.vector_store_path}")
    print(f"Embedding model: {settings.embedding_model}\n")

    rag = get_rag_pipeline()

    try:
        count = rag.ingest_documents(settings.documents_path)
        print(f"\n[OK] Successfully indexed {count} document chunks.")
        print("     You can now start the server:  uvicorn app.main:app --reload\n")
    except FileNotFoundError as exc:
        print(f"\n[ERR] {exc}\n")
        sys.exit(1)
    except ValueError as exc:
        print(f"\n[ERR] {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

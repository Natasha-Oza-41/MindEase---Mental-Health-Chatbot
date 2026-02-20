"""
RAG (Retrieval-Augmented Generation) Pipeline

Flow:
    1. Documents are chunked and embedded using a local sentence-transformer model.
    2. Embeddings are stored in ChromaDB (persisted to disk).
    3. At query time, the user message is embedded and the top-K most
       semantically similar chunks are retrieved.
    4. Retrieved chunks are injected into the LLM system prompt as context.

Why this approach:
    - Grounds the LLM in verified mental health content, reducing hallucination.
    - Local embeddings = zero API cost for the retrieval step.
    - ChromaDB persists to disk — no server needed, works offline after first setup.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)

COLLECTION_NAME = "mental_health_docs"


class RAGPipeline:
    def __init__(self) -> None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self._encoder = SentenceTransformer(settings.embedding_model)

        self._client = chromadb.PersistentClient(path=settings.vector_store_path)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._ready = self._collection.count() > 0
        if self._ready:
            logger.info(
                f"RAG pipeline ready — {self._collection.count()} chunks indexed."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_documents(self, docs_path: str) -> int:
        """
        Read all .md files from docs_path, chunk them, embed, and upsert
        into ChromaDB. Safe to re-run (upsert is idempotent).
        """
        docs_dir = Path(docs_path)
        if not docs_dir.exists():
            raise FileNotFoundError(
                f"Documents directory not found: {docs_path}\n"
                "Make sure the path in DOCUMENTS_PATH (.env) is correct."
            )

        documents, metadatas, ids = [], [], []
        doc_id = 0

        for file_path in sorted(docs_dir.glob("*.md")):
            content = file_path.read_text(encoding="utf-8")
            chunks = self._chunk_text(content, chunk_size=400, overlap=60)
            for chunk in chunks:
                if len(chunk.strip()) > 20:  # skip near-empty chunks
                    documents.append(chunk)
                    metadatas.append({"source": file_path.name})
                    ids.append(f"chunk_{doc_id}")
                    doc_id += 1

        if not documents:
            raise ValueError("No content found in documents directory.")

        logger.info(f"Embedding {len(documents)} chunks…")
        embeddings = self._encoder.encode(documents, show_progress_bar=True).tolist()

        self._collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        self._ready = True
        logger.info(f"Ingested {len(documents)} chunks successfully.")
        return len(documents)

    def retrieve(self, query: str, top_k: int | None = None) -> List[Tuple[str, str]]:
        """
        Return up to top_k (document_chunk, source_filename) tuples
        most relevant to the query.
        """
        if not self._ready:
            return []

        k = min(top_k or settings.rag_top_k, self._collection.count())
        if k == 0:
            return []

        query_embedding = self._encoder.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        )

        pairs: List[Tuple[str, str]] = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            pairs.append((doc, meta.get("source", "unknown")))
        return pairs

    def is_ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chunk_text(
        self, text: str, chunk_size: int = 400, overlap: int = 60
    ) -> List[str]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        step = max(chunk_size - overlap, 1)
        return [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), step)
        ]


# --- Singleton ---
_rag_instance: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGPipeline()
    return _rag_instance

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LLM ---
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 500

    # --- RAG ---
    vector_store_path: str = "./vector_store"
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 3
    documents_path: str = "./data/mental_health_docs"

    # --- Memory ---
    max_conversation_history: int = 10

    # --- App ---
    debug: bool = False
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "case_sensitive": False}


@lru_cache()
def get_settings() -> Settings:
    return Settings()

"""
LLM Client abstraction layer.

Using an abstract base class means you can swap OpenAI for any compatible
API (Groq, Azure OpenAI, Ollama with OpenAI-compatible endpoint) without
touching the rest of the codebase.
"""
from abc import ABC, abstractmethod
from typing import Dict, List

import openai

from app.config import get_settings

settings = get_settings()


class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send a list of messages and return the assistant reply."""
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    def __init__(self) -> None:
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

    def chat(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except openai.AuthenticationError as exc:
            raise ValueError(
                "Invalid OpenAI API key. Please check OPENAI_API_KEY in your .env file."
            ) from exc
        except openai.RateLimitError as exc:
            raise RuntimeError(
                "OpenAI rate limit reached. Please wait a moment and try again."
            ) from exc
        except openai.APIConnectionError as exc:
            raise RuntimeError(
                "Cannot reach the OpenAI API. Check your internet connection."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc


# --- Singleton ---
_llm_instance: BaseLLMClient | None = None


def get_llm_client() -> BaseLLMClient:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = OpenAIClient()
    return _llm_instance

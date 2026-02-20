"""
In-memory conversation store, keyed by session_id.

Design notes:
- Uses a plain Python dict — simple and sufficient for a local demo.
- Production upgrade path: swap with Redis or a database-backed store.
- A sliding window prevents the LLM context window from overflowing:
  we keep the last (max_history * 2) messages (each turn = user + assistant).
"""
from collections import defaultdict
from typing import Dict, List, Optional

from app.config import get_settings

settings = get_settings()


class ConversationMemory:
    def __init__(self) -> None:
        # {session_id: [{"role": "user"|"assistant", "content": "..."}]}
        self._sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self._window = settings.max_conversation_history * 2  # messages, not turns

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._sessions[session_id].append({"role": role, "content": content})
        # Trim to sliding window
        if len(self._sessions[session_id]) > self._window:
            self._sessions[session_id] = self._sessions[session_id][-self._window :]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self._sessions[session_id])  # return a copy

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        return len(self._sessions)


# --- Singleton ---
_memory_instance: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance

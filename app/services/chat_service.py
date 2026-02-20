"""
ChatService — the main orchestration layer.

Request flow:
    1. Crisis detection  (always runs first — safety gate)
    2. If CRITICAL/HIGH  → return pre-written crisis response + resources
    3. RAG retrieval     → fetch relevant mental health context
    4. Prompt building   → system prompt + RAG context + conversation history
    5. LLM call          → generate empathetic response
    6. Safety disclaimer → appended for LOW/MEDIUM crisis signals
    7. Memory update     → save turn to session history
    8. Conversation log  → persist to JSONL file
"""
from typing import Dict, List

from app.config import get_settings
from app.core.crisis_detector import CrisisDetector, get_crisis_detector
from app.core.llm_client import BaseLLMClient, get_llm_client
from app.core.memory import ConversationMemory, get_memory
from app.core.rag_pipeline import RAGPipeline, get_rag_pipeline
from app.models.schemas import ChatRequest, ChatResponse, CrisisSeverity
from app.utils.logger import log_conversation, setup_logger

settings = get_settings()
logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt — defines the persona and guardrails for the LLM
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a compassionate and knowledgeable mental health support assistant.

Your purpose:
- Listen actively and respond with empathy and without judgment.
- Provide evidence-based psychoeducation and coping strategies.
- Validate the user's feelings and help them feel heard.
- Encourage professional help when the situation warrants it.

Strict boundaries:
- You are NOT a therapist or licensed clinician.
- Do NOT diagnose any mental health condition.
- Do NOT prescribe or recommend specific medications.
- If someone is in crisis, direct them to emergency services immediately.
- Always remind users you are an AI when appropriate.

Tone: Warm, calm, non-judgmental. Use plain language. Avoid clinical jargon.
Length: Keep responses concise (3-5 sentences unless the user asks for more detail).
"""

_SAFETY_DISCLAIMER = (
    "\n\n---\n"
    "*Remember: I'm an AI assistant, not a licensed therapist. "
    "If you're struggling, please consider reaching out to a mental health "
    "professional. The Crisis Text Line is available 24/7 — text HOME to 741741.*"
)


class ChatService:
    def __init__(
        self,
        llm: BaseLLMClient,
        memory: ConversationMemory,
        crisis_detector: CrisisDetector,
        rag: RAGPipeline,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.crisis_detector = crisis_detector
        self.rag = rag

    def process(self, request: ChatRequest) -> ChatResponse:
        sid = request.session_id
        user_msg = request.message

        # ── Step 1: Crisis detection ──────────────────────────────────────
        crisis_info = self.crisis_detector.detect(user_msg)
        logger.info(f"[{sid}] Crisis severity: {crisis_info.severity}")

        # ── Step 2: Hard override for CRITICAL / HIGH ─────────────────────
        if self.crisis_detector.requires_override(crisis_info.severity):
            safe_response = self.crisis_detector.get_crisis_response(crisis_info.severity)
            self._save_turn(sid, user_msg, safe_response, crisis_info.severity)
            return ChatResponse(
                session_id=sid,
                response=safe_response,
                crisis_info=crisis_info,
            )

        # ── Step 3: RAG retrieval ─────────────────────────────────────────
        retrieved = self.rag.retrieve(user_msg)
        sources = list({src for _, src in retrieved})
        context_block = self._format_context(retrieved)

        # ── Step 4: Build message list for LLM ───────────────────────────
        messages = self._build_messages(sid, user_msg, context_block)

        # ── Step 5: LLM call ──────────────────────────────────────────────
        try:
            bot_response = self.llm.chat(messages)
        except Exception as exc:
            logger.error(f"[{sid}] LLM error: {exc}")
            bot_response = (
                "I'm having trouble connecting right now. "
                "If you need immediate support, please call or text 988 (US) "
                "or text HOME to 741741."
            )

        # ── Step 6: Append safety disclaimer for LOW / MEDIUM signals ─────
        if crisis_info.severity in (CrisisSeverity.LOW, CrisisSeverity.MEDIUM):
            bot_response += _SAFETY_DISCLAIMER

        # ── Step 7 & 8: Persist ───────────────────────────────────────────
        self._save_turn(sid, user_msg, bot_response, crisis_info.severity)

        return ChatResponse(
            session_id=sid,
            response=bot_response,
            crisis_info=crisis_info if crisis_info.severity != CrisisSeverity.NONE else None,
            sources=sources or None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_turn(
        self,
        sid: str,
        user_msg: str,
        bot_response: str,
        severity: CrisisSeverity,
    ) -> None:
        self.memory.add_message(sid, "user", user_msg)
        self.memory.add_message(sid, "assistant", bot_response)
        log_conversation(sid, user_msg, bot_response, severity.value)

    def _format_context(self, retrieved: list) -> str:
        if not retrieved:
            return ""
        lines = ["### Relevant information from verified mental health resources:\n"]
        for doc, source in retrieved:
            lines.append(f"**[{source}]**\n{doc}\n")
        return "\n".join(lines)

    def _build_messages(
        self, sid: str, user_msg: str, context: str
    ) -> List[Dict[str, str]]:
        system_content = _SYSTEM_PROMPT
        if context:
            system_content += f"\n\n{context}"

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
        messages.extend(self.memory.get_history(sid))
        messages.append({"role": "user", "content": user_msg})
        return messages


# --- Factory (used by FastAPI Depends) ---
def get_chat_service() -> ChatService:
    return ChatService(
        llm=get_llm_client(),
        memory=get_memory(),
        crisis_detector=get_crisis_detector(),
        rag=get_rag_pipeline(),
    )

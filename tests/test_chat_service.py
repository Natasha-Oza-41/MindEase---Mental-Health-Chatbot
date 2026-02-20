"""
Unit tests for the ChatService orchestration layer.

Uses mocks to isolate the service from external dependencies (LLM API, ChromaDB).

Run with:  pytest tests/test_chat_service.py -v
"""
from unittest.mock import MagicMock

import pytest

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    CrisisInfo,
    CrisisResource,
    CrisisSeverity,
)
from app.services.chat_service import ChatService


def _make_crisis_info(severity: CrisisSeverity) -> CrisisInfo:
    resources = (
        [CrisisResource(name="Test Line", contact="000", available="24/7", url="http://example.com")]
        if severity != CrisisSeverity.NONE
        else []
    )
    return CrisisInfo(severity=severity, detected_keywords=[], resources=resources)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat.return_value = "I hear you. That sounds really difficult."
    return llm


@pytest.fixture
def mock_memory():
    memory = MagicMock()
    memory.get_history.return_value = []
    return memory


@pytest.fixture
def mock_detector():
    detector = MagicMock()
    detector.detect.return_value = _make_crisis_info(CrisisSeverity.NONE)
    detector.requires_override.return_value = False
    return detector


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.retrieve.return_value = []
    return rag


@pytest.fixture
def service(mock_llm, mock_memory, mock_detector, mock_rag) -> ChatService:
    return ChatService(
        llm=mock_llm,
        memory=mock_memory,
        crisis_detector=mock_detector,
        rag=mock_rag,
    )


class TestNormalFlow:
    def test_returns_chat_response(self, service):
        req = ChatRequest(session_id="sess-1", message="I've been feeling anxious.")
        resp = service.process(req)
        assert isinstance(resp, ChatResponse)
        assert resp.session_id == "sess-1"
        assert len(resp.response) > 0

    def test_no_crisis_info_when_none(self, service):
        req = ChatRequest(session_id="sess-1", message="How can I reduce stress?")
        resp = service.process(req)
        assert resp.crisis_info is None

    def test_llm_is_called(self, service, mock_llm):
        req = ChatRequest(session_id="sess-2", message="Tell me about mindfulness.")
        service.process(req)
        mock_llm.chat.assert_called_once()

    def test_memory_is_updated(self, service, mock_memory):
        req = ChatRequest(session_id="sess-3", message="I feel overwhelmed.")
        service.process(req)
        assert mock_memory.add_message.call_count == 2  # user + assistant

    def test_rag_is_queried(self, service, mock_rag):
        req = ChatRequest(session_id="sess-4", message="What is anxiety?")
        service.process(req)
        mock_rag.retrieve.assert_called_once()


class TestCrisisOverride:
    def test_critical_overrides_llm(self, service, mock_detector, mock_llm):
        mock_detector.detect.return_value = _make_crisis_info(CrisisSeverity.CRITICAL)
        mock_detector.requires_override.return_value = True
        mock_detector.get_crisis_response.return_value = "Please call 988 immediately."

        req = ChatRequest(session_id="crisis-sess", message="I want to end my life.")
        resp = service.process(req)

        mock_llm.chat.assert_not_called()  # LLM should NOT be called
        assert resp.crisis_info is not None
        assert resp.crisis_info.severity == CrisisSeverity.CRITICAL
        assert "988" in resp.response

    def test_crisis_info_attached_to_response(self, service, mock_detector):
        mock_detector.detect.return_value = _make_crisis_info(CrisisSeverity.HIGH)
        mock_detector.requires_override.return_value = True
        mock_detector.get_crisis_response.return_value = "Please reach out to a crisis line."

        req = ChatRequest(session_id="high-crisis", message="I feel hopeless.")
        resp = service.process(req)

        assert resp.crisis_info is not None
        assert resp.crisis_info.severity == CrisisSeverity.HIGH


class TestLLMFailover:
    def test_llm_error_returns_safe_fallback(self, service, mock_llm):
        mock_llm.chat.side_effect = RuntimeError("Connection failed")

        req = ChatRequest(session_id="err-sess", message="I'm feeling down.")
        resp = service.process(req)

        # Should not raise; should return a safe fallback message
        assert "988" in resp.response or "141741" in resp.response or "support" in resp.response.lower()

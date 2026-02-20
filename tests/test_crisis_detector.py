"""
Unit tests for the crisis detection layer.

Run with:  pytest tests/test_crisis_detector.py -v
"""
import pytest

from app.core.crisis_detector import CrisisDetector
from app.models.schemas import CrisisSeverity


@pytest.fixture
def detector() -> CrisisDetector:
    return CrisisDetector()


class TestCrisisDetection:
    def test_no_crisis_everyday_message(self, detector):
        result = detector.detect("I've been feeling a bit stressed about my exams lately.")
        assert result.severity == CrisisSeverity.NONE
        assert result.detected_keywords == []
        assert result.resources == []

    def test_critical_severity_detected(self, detector):
        result = detector.detect("I've been thinking about suicide and I don't know what to do.")
        assert result.severity == CrisisSeverity.CRITICAL
        assert len(result.resources) > 0
        assert "suicide" in result.detected_keywords

    def test_high_severity_detected(self, detector):
        result = detector.detect("I feel completely hopeless, like there's nothing to live for anymore.")
        assert result.severity in (CrisisSeverity.HIGH, CrisisSeverity.CRITICAL)
        assert len(result.resources) > 0

    def test_medium_severity_detected(self, detector):
        result = detector.detect("I hate myself and feel like I'm falling apart.")
        assert result.severity in (
            CrisisSeverity.MEDIUM, CrisisSeverity.HIGH, CrisisSeverity.CRITICAL
        )

    def test_low_severity_detected(self, detector):
        result = detector.detect("I've been feeling depressed and can't sleep for days.")
        assert result.severity != CrisisSeverity.NONE

    def test_case_insensitive_matching(self, detector):
        result = detector.detect("I WANT TO KILL MYSELF.")
        assert result.severity == CrisisSeverity.CRITICAL

    def test_multiple_keywords_deduplicated(self, detector):
        result = detector.detect("I want to die, I want to die, I want to die.")
        # Keywords should be deduplicated in the list
        assert len(result.detected_keywords) == len(set(result.detected_keywords))


class TestOverrideLogic:
    def test_critical_requires_override(self, detector):
        assert detector.requires_override(CrisisSeverity.CRITICAL) is True

    def test_high_requires_override(self, detector):
        assert detector.requires_override(CrisisSeverity.HIGH) is True

    def test_medium_does_not_require_override(self, detector):
        assert detector.requires_override(CrisisSeverity.MEDIUM) is False

    def test_low_does_not_require_override(self, detector):
        assert detector.requires_override(CrisisSeverity.LOW) is False

    def test_none_does_not_require_override(self, detector):
        assert detector.requires_override(CrisisSeverity.NONE) is False


class TestCrisisResponse:
    def test_critical_response_mentions_988(self, detector):
        response = detector.get_crisis_response(CrisisSeverity.CRITICAL)
        assert "988" in response
        assert len(response) > 50

    def test_high_response_mentions_crisis_line(self, detector):
        response = detector.get_crisis_response(CrisisSeverity.HIGH)
        assert "988" in response or "741741" in response

    def test_none_returns_empty_string(self, detector):
        response = detector.get_crisis_response(CrisisSeverity.NONE)
        assert response == ""

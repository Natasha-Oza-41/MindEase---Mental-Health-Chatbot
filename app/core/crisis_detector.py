"""
Crisis Detection Layer

Strategy:
    1. Keyword matching — fast, deterministic, transparent.
    2. Severity tiering — CRITICAL/HIGH trigger a hard override of the LLM
       response and surface emergency resources immediately.
    3. MEDIUM/LOW allow the LLM to respond but append a safety disclaimer.

This is intentionally conservative. False positives (flagging non-crisis
messages) are far less harmful than false negatives (missing a real crisis).

NOTE: This layer is a safety net, NOT a clinical assessment tool.
"""
from typing import Dict, List

from app.models.schemas import CrisisInfo, CrisisResource, CrisisSeverity

# ---------------------------------------------------------------------------
# Keyword taxonomy by severity
# ---------------------------------------------------------------------------
CRISIS_KEYWORDS: Dict[CrisisSeverity, List[str]] = {
    CrisisSeverity.CRITICAL: [
        "kill myself",
        "end my life",
        "want to die",
        "suicide",
        "suicidal",
        "take my life",
        "end it all",
        "no reason to live",
        "better off dead",
        "don't want to be alive",
        "self-harm",
        "hurt myself",
        "cutting myself",
        "overdose",
        "harm myself",
    ],
    CrisisSeverity.HIGH: [
        "hopeless",
        "worthless",
        "can't go on",
        "give up on life",
        "nothing to live for",
        "disappear forever",
        "everyone would be better without me",
        "i can't do this anymore",
        "no way out",
    ],
    CrisisSeverity.MEDIUM: [
        "i hate myself",
        "i'm a burden",
        "can't take it anymore",
        "no point in anything",
        "completely alone",
        "no one cares about me",
        "falling apart",
        "losing my mind",
        "breaking down",
        "feel empty inside",
    ],
    CrisisSeverity.LOW: [
        "depressed",
        "very sad",
        "extremely anxious",
        "panic attack",
        "can't sleep for days",
        "stopped eating",
        "feel numb",
        "feel hopeless",
    ],
}

# ---------------------------------------------------------------------------
# Crisis resources (real, publicly available)
# ---------------------------------------------------------------------------
CRISIS_RESOURCES: List[CrisisResource] = [
    CrisisResource(
        name="988 Suicide & Crisis Lifeline",
        contact="Call or text 988 (US)",
        available="24/7",
        url="https://988lifeline.org",
    ),
    CrisisResource(
        name="Crisis Text Line",
        contact="Text HOME to 741741",
        available="24/7",
        url="https://www.crisistextline.org",
    ),
    CrisisResource(
        name="SAMHSA National Helpline",
        contact="1-800-662-4357",
        available="24/7, free, confidential",
        url="https://www.samhsa.gov/find-help/national-helpline",
    ),
    CrisisResource(
        name="International Crisis Centers",
        contact="Find your local center",
        available="Varies by country",
        url="https://www.iasp.info/resources/Crisis_Centres/",
    ),
]

# Severity ordering from highest to lowest (for first-match logic)
_SEVERITY_ORDER = [
    CrisisSeverity.CRITICAL,
    CrisisSeverity.HIGH,
    CrisisSeverity.MEDIUM,
    CrisisSeverity.LOW,
]


class CrisisDetector:
    def detect(self, text: str) -> CrisisInfo:
        """Scan text and return a CrisisInfo with severity + matched keywords."""
        text_lower = text.lower()
        detected: List[str] = []
        highest = CrisisSeverity.NONE

        for severity in _SEVERITY_ORDER:
            for keyword in CRISIS_KEYWORDS[severity]:
                if keyword in text_lower:
                    detected.append(keyword)
                    if highest == CrisisSeverity.NONE:
                        highest = severity  # capture only the worst tier

        resources = CRISIS_RESOURCES if highest != CrisisSeverity.NONE else []
        return CrisisInfo(
            severity=highest,
            detected_keywords=list(set(detected)),
            resources=resources,
        )

    def requires_override(self, severity: CrisisSeverity) -> bool:
        """Return True when the LLM response should be replaced by crisis messaging."""
        return severity in (CrisisSeverity.CRITICAL, CrisisSeverity.HIGH)

    def get_crisis_response(self, severity: CrisisSeverity) -> str:
        """Return a safe, pre-written crisis response appropriate for the severity."""
        if severity == CrisisSeverity.CRITICAL:
            return (
                "I'm very concerned about what you've just shared, and I want you to "
                "know that your life has value. Please reach out for immediate support "
                "right now — you do not have to face this alone.\n\n"
                "If you are in the US, call or text **988** (Suicide & Crisis Lifeline) "
                "— available 24/7, free, and confidential. You can also text HOME to "
                "**741741** (Crisis Text Line) to chat with a trained counselor.\n\n"
                "If you are in immediate danger, please call emergency services (911 in the US).\n\n"
                "I am an AI and cannot provide the level of support you deserve right now. "
                "A real human counselor is waiting to help."
            )
        if severity == CrisisSeverity.HIGH:
            return (
                "It sounds like you're carrying something very heavy right now, and I "
                "hear you. What you're feeling is real, and you deserve support from "
                "someone equipped to help.\n\n"
                "Please consider reaching out to a crisis counselor — they are trained "
                "for exactly this moment. You can call or text **988** anytime, or "
                "text HOME to **741741** to connect via text. Both are free and confidential.\n\n"
                "I'm here to listen, but please also connect with a human who can truly support you."
            )
        return ""


# --- Singleton ---
_detector_instance: CrisisDetector | None = None


def get_crisis_detector() -> CrisisDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = CrisisDetector()
    return _detector_instance

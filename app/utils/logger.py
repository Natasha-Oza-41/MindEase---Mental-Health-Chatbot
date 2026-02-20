"""
Logging utilities.

Two log streams:
    1. app_YYYYMMDD.log  — general application log (INFO/ERROR/etc.)
    2. conversations_YYYYMMDD.jsonl — one JSON line per conversation turn,
       making it easy to analyze or fine-tune on later.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured; avoid duplicate handlers

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(
        LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log",
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def log_conversation(
    session_id: str,
    user_message: str,
    bot_response: str,
    crisis_severity: str = "none",
    sources: list | None = None,
) -> None:
    """Append a single conversation turn to today's JSONL conversation log."""
    log_file = LOG_DIR / f"conversations_{datetime.now().strftime('%Y%m%d')}.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "user_message": user_message,
        "bot_response": bot_response,
        "crisis_severity": crisis_severity,
        "sources": sources or [],
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.security.guardrails import redact_pii


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "rag_service.jsonl"

LOGGER_NAME = "rag_service"

SENSITIVE_FIELD_NAMES = {
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
}


class JsonFormatter(logging.Formatter):
    """Format each log record as one JSON object."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "event": getattr(
                record,
                "event_name",
                record.getMessage(),
            ),
            "details": getattr(
                record,
                "event_details",
                {},
            ),
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(
            log_entry,
            ensure_ascii=False,
        )


def sanitise_log_value(value, field_name=None):
    """Redact PII and secret fields before logging."""
    if (
        field_name
        and field_name.lower() in SENSITIVE_FIELD_NAMES
    ):
        return "[REDACTED SECRET]"

    if isinstance(value, str):
        return redact_pii(value)

    if isinstance(value, dict):
        return {
            key: sanitise_log_value(
                item,
                field_name=key,
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            sanitise_log_value(item)
            for item in value
        ]

    return value


def get_service_logger():
    """Create and return the shared service logger."""
    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        return logger

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = JsonFormatter()

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_event(level, event_name, **details):
    """Write one structured and sanitised log event."""
    logger = get_service_logger()

    safe_details = sanitise_log_value(details)

    numeric_level = getattr(
        logging,
        level.upper(),
        logging.INFO,
    )

    logger.log(
        numeric_level,
        event_name,
        extra={
            "event_name": event_name,
            "event_details": safe_details,
        },
    )


if __name__ == "__main__":
    log_event(
        "INFO",
        "logger_test",
        question=(
            "Contact alex@example.com "
            "or call +44 7700 900123."
        ),
        latency_ms=3.42,
        api_key="this-must-not-appear",
    )

    print(f"Test log written to: {LOG_FILE}")
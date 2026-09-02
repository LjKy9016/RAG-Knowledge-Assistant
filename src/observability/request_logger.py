from src.observability.logger import log_event


def convert_session_id(session_id):
    """Convert a session ID into a JSON-safe value."""
    if session_id is None:
        return None

    return str(session_id)


def log_request_received(
    request_id,
    question,
    session_id,
):
    """Log the beginning of an API request."""
    log_event(
        "INFO",
        "request_received",
        request_id=request_id,
        session_id=convert_session_id(session_id),
        question=question,
    )


def log_request_completed(
    request_id,
    question,
    result,
    latency_ms,
):
    """Log a completed answer request."""
    metadata = result.get("_metadata", {})

    sources = [
        {
            "file_name": source["file_name"],
            "page_number": source["page_number"],
        }
        for source in result.get("sources", [])
    ]

    log_event(
        "INFO",
        "request_completed",
        request_id=request_id,
        session_id=convert_session_id(
            result.get("session_id")
        ),
        question=question,
        latency_ms=round(latency_ms, 2),
        sources=sources,
        **metadata,
    )


def log_request_failed(
    request_id,
    question,
    session_id,
    latency_ms,
    error,
    status_code,
):
    """Log a failed answer request without exposing secrets."""
    log_event(
        "ERROR",
        "request_failed",
        request_id=request_id,
        session_id=convert_session_id(session_id),
        question=question,
        latency_ms=round(latency_ms, 2),
        status_code=status_code,
        error_type=type(error).__name__,
    )


def log_session_deleted(
    session_id,
    deleted,
):
    """Log a session deletion request."""
    log_event(
        "INFO",
        "session_deleted",
        session_id=str(session_id),
        deleted=deleted,
    )
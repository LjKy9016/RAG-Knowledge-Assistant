from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    AskRequest,
    AskResponse,
    SourceResponse,
)


def test_accepts_valid_ask_request():
    request = AskRequest(
        question="What is the API rate limit?"
    )

    assert request.question == "What is the API rate limit?"
    assert request.session_id is None


def test_parses_session_id_string():
    session_id = uuid4()

    request = AskRequest(
        question="Follow-up question",
        session_id=str(session_id),
    )

    assert request.session_id == session_id


def test_rejects_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(question="")


def test_rejects_question_over_limit():
    with pytest.raises(ValidationError):
        AskRequest(question="a" * 2001)


def test_builds_response_schema():
    session_id = uuid4()
    source = SourceResponse(
        source_number=1,
        file_name="document.pdf",
        page_number=1,
    )

    response = AskResponse(
        session_id=session_id,
        answer="Test answer",
        sources=[source],
        latency_ms=10.5,
    )

    assert response.session_id == session_id
    assert response.sources[0].file_name == "document.pdf"
    assert response.latency_ms == 10.5
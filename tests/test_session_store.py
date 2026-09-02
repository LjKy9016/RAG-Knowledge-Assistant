import pytest

from src.generation.session_store import (
    MAX_HISTORY_MESSAGES,
    add_message,
    delete_session,
    get_history,
    get_or_create_session,
)


def test_creates_and_deletes_session():
    session_id = get_or_create_session()

    try:
        assert get_history(session_id) == []
    finally:
        assert delete_session(session_id) is True

    assert delete_session(session_id) is False


def test_reuses_existing_session_id():
    session_id = get_or_create_session()

    try:
        returned_id = get_or_create_session(str(session_id))
        assert returned_id == session_id
    finally:
        delete_session(session_id)


def test_adds_messages_to_history():
    session_id = get_or_create_session()

    try:
        add_message(session_id, "user", "Question")
        add_message(session_id, "assistant", "Answer")

        assert get_history(session_id) == [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    finally:
        delete_session(session_id)


def test_rejects_invalid_role():
    session_id = get_or_create_session()

    try:
        with pytest.raises(ValueError):
            add_message(session_id, "system", "Message")
    finally:
        delete_session(session_id)


def test_rejects_empty_message():
    session_id = get_or_create_session()

    try:
        with pytest.raises(ValueError):
            add_message(session_id, "user", "   ")
    finally:
        delete_session(session_id)


def test_limits_history_length():
    session_id = get_or_create_session()

    try:
        for index in range(MAX_HISTORY_MESSAGES + 2):
            add_message(
                session_id,
                "user",
                f"Message {index}",
            )

        history = get_history(session_id)

        assert len(history) == MAX_HISTORY_MESSAGES
        assert history[0]["content"] == "Message 2"
    finally:
        delete_session(session_id)
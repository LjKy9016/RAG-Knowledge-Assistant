from threading import Lock
from uuid import UUID, uuid4

# python -m src.generation.session_store

MAX_HISTORY_MESSAGES = 10

# This in-memory store is suitable for the local MVP.
sessions = {}
sessions_lock = Lock()


def get_or_create_session(session_id=None):
    """Return an existing session or create a new one."""
    if session_id is None:
        session_id = uuid4()

    if isinstance(session_id, str):
        session_id = UUID(session_id)

    session_key = str(session_id)

    with sessions_lock:
        if session_key not in sessions:
            sessions[session_key] = []

    return session_id


def get_history(session_id):
    """Return a copy of the conversation history."""
    session_key = str(session_id)

    with sessions_lock:
        history = sessions.get(session_key, [])
        return [message.copy() for message in history]


def add_message(session_id, role, content):
    """Add one user or assistant message to a session."""
    if role not in {"user", "assistant"}:
        raise ValueError(
            "Role must be either 'user' or 'assistant'"
        )

    if not content or not content.strip():
        raise ValueError("Message content cannot be empty")

    session_key = str(session_id)

    with sessions_lock:
        if session_key not in sessions:
            sessions[session_key] = []

        sessions[session_key].append(
            {
                "role": role,
                "content": content.strip(),
            }
        )

        # Keep only the most recent five user-assistant turns.
        sessions[session_key] = sessions[session_key][
            -MAX_HISTORY_MESSAGES:
        ]


def delete_session(session_id):
    """Delete a session and return whether it existed."""
    session_key = str(session_id)

    with sessions_lock:
        if session_key in sessions:
            del sessions[session_key]
            return True

    return False


def get_session_count():
    """Return the number of sessions currently stored."""
    with sessions_lock:
        return len(sessions)


if __name__ == "__main__":
    test_session_id = get_or_create_session()

    add_message(
        test_session_id,
        "user",
        "How many days of annual leave do employees receive?",
    )

    add_message(
        test_session_id,
        "assistant",
        "Full-time employees receive 25 days.",
    )

    print(f"Session ID: {test_session_id}")
    print(f"History: {get_history(test_session_id)}")
    print(f"Session count: {get_session_count()}")

    was_deleted = delete_session(test_session_id)

    print(f"Deleted: {was_deleted}")
    print(f"Session count after deletion: {get_session_count()}")
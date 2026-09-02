from uuid import UUID

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Data received by the question-answering endpoint."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural-language question from the user.",
    )

    session_id: UUID | None = Field(
        default=None,
        description=(
            "Existing conversation session ID. "
            "Leave empty to start a new session."
        ),
    )


class SourceResponse(BaseModel):
    """One document source used by the RAG response."""

    source_number: int
    file_name: str
    page_number: int


class AskResponse(BaseModel):
    """Data returned by the question-answering endpoint."""

    session_id: UUID
    answer: str
    sources: list[SourceResponse]
    latency_ms: float


class HealthResponse(BaseModel):
    """Service health information."""

    status: str
    vector_store_ready: bool


class DeleteSessionResponse(BaseModel):
    """Result returned after clearing a conversation session."""

    session_id: UUID
    deleted: bool
    message: str
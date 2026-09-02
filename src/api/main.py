from contextlib import asynccontextmanager
from functools import partial
from time import perf_counter
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from src.api.schemas import (
    AskRequest,
    AskResponse,
    DeleteSessionResponse,
    HealthResponse,
)
from src.generation.answer_generator import generate_answer
from src.generation.session_store import delete_session
from src.observability.request_logger import (
    log_request_completed,
    log_request_failed,
    log_request_received,
    log_session_deleted,
)
from src.retrieval.vector_store import (
    build_vector_store,
    get_collection,
)

# uvicorn src.api.main:app --reload
# http://127.0.0.1:8000/docs

@asynccontextmanager
async def lifespan(app):
    """Prepare the vector store when the API starts."""
    collection = get_collection()

    if collection.count() == 0:
        print("Vector store is empty. Building it now...")

        await run_in_threadpool(
            build_vector_store
        )

    yield


app = FastAPI(
    title="Northstar RAG Knowledge Assistant",
    description=(
        "A bilingual retrieval-augmented question-answering "
        "service for internal company documents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health_check():
    """Check whether the API and vector store are ready."""
    collection = get_collection()
    vector_store_ready = collection.count() > 0

    return {
        "status": (
            "ok"
            if vector_store_ready
            else "not_ready"
        ),
        "vector_store_ready": vector_store_ready,
    }


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["Question Answering"],
)
async def ask_question(request: AskRequest):
    """Answer a question using retrieved document context."""
    request_id = str(uuid4())
    start_time = perf_counter()

    log_request_received(
        request_id=request_id,
        question=request.question,
        session_id=request.session_id,
    )

    try:
        generation_task = partial(
            generate_answer,
            question=request.question,
            session_id=request.session_id,
        )

        result = await run_in_threadpool(
            generation_task
        )

    except ValueError as error:
        latency_ms = (
            perf_counter() - start_time
        ) * 1000

        log_request_failed(
            request_id=request_id,
            question=request.question,
            session_id=request.session_id,
            latency_ms=latency_ms,
            error=error,
            status_code=400,
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        latency_ms = (
            perf_counter() - start_time
        ) * 1000

        log_request_failed(
            request_id=request_id,
            question=request.question,
            session_id=request.session_id,
            latency_ms=latency_ms,
            error=error,
            status_code=500,
        )

        raise HTTPException(
            status_code=500,
            detail="The answer could not be generated.",
        ) from error

    latency_ms = (
        perf_counter() - start_time
    ) * 1000

    log_request_completed(
        request_id=request_id,
        question=request.question,
        result=result,
        latency_ms=latency_ms,
    )

    return {
        "session_id": result["session_id"],
        "answer": result["answer"],
        "sources": result["sources"],
        "latency_ms": round(latency_ms, 2),
    }


@app.delete(
    "/sessions/{session_id}",
    response_model=DeleteSessionResponse,
    tags=["Sessions"],
)
def clear_session(session_id: UUID):
    """Delete the stored history for one session."""
    was_deleted = delete_session(session_id)

    log_session_deleted(
        session_id=session_id,
        deleted=was_deleted,
    )

    if was_deleted:
        message = "Session history was deleted."
    else:
        message = "Session was not found."

    return {
        "session_id": session_id,
        "deleted": was_deleted,
        "message": message,
    }
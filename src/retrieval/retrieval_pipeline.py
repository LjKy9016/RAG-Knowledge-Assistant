from src.config import (
    DEFAULT_TOP_K,
    RERANK_CANDIDATE_K,
    USE_RERANKER,
)
from src.retrieval.relevance import filter_relevant_results
from src.retrieval.reranker import rerank_results
from src.retrieval.vector_store import search_chunks


def retrieve_relevant_chunks(
    query,
    top_k=DEFAULT_TOP_K,
    use_reranker=None,
):
    """Retrieve, filter and optionally rerank document chunks."""
    if not query or not query.strip():
        raise ValueError("Retrieval query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if use_reranker is None:
        reranker_enabled = USE_RERANKER
    else:
        reranker_enabled = use_reranker

    if reranker_enabled:
        candidate_k = max(
            RERANK_CANDIDATE_K,
            top_k,
        )
    else:
        candidate_k = top_k

    search_results = search_chunks(
        query,
        top_k=candidate_k,
    )

    search_results = filter_relevant_results(
        search_results
    )

    if reranker_enabled:
        return rerank_results(
            query,
            search_results,
            top_k=top_k,
        )

    return search_results[:top_k]
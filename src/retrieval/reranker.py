from sentence_transformers import CrossEncoder

from src.retrieval.relevance import filter_relevant_results
from src.retrieval.vector_store import search_chunks

# python -m src.retrieval.reranker

RERANKER_MODEL_NAME = (
    "amber-tech/"
    "bert-multilingual-passage-reranking-msmarco"
)

reranker_model = None


def get_reranker_model():
    """Load and return the multilingual reranker model."""
    global reranker_model

    if reranker_model is None:
        print(
            f"Loading reranker model: "
            f"{RERANKER_MODEL_NAME}"
        )

        reranker_model = CrossEncoder(
            RERANKER_MODEL_NAME,
            max_length=512,
        )

        print("Reranker model loaded")

    return reranker_model


def rerank_results(query, search_results, top_k=3):
    """Rerank retrieved chunks using query-passage scores."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not search_results:
        return []

    model = get_reranker_model()

    query_document_pairs = [
        (query, result["text"])
        for result in search_results
    ]

    scores = model.predict(
        query_document_pairs,
        show_progress_bar=False,
    )

    reranked_results = []

    for result, score in zip(search_results, scores):
        reranked_result = result.copy()

        # This model returns [not relevant, relevant] scores.
        if getattr(score, "ndim", 0) > 0:
            score_value = float(score[-1])
        else:
            score_value = float(score)

        reranked_result["reranker_score"] = score_value
        reranked_results.append(reranked_result)

    reranked_results.sort(
        key=lambda result: result["reranker_score"],
        reverse=True,
    )

    return reranked_results[:top_k]


if __name__ == "__main__":
    test_query = "伦敦酒店每晚的报销限额是多少？"

    initial_results = search_chunks(
        test_query,
        top_k=6,
    )

    relevant_results = filter_relevant_results(
        initial_results
    )

    print("\nBefore reranking:")

    for position, result in enumerate(
        relevant_results,
        start=1,
    ):
        print(
            f"{position}. {result['file_name']} | "
            f"chunk {result['chunk_index']} | "
            f"distance {result['distance']:.4f}"
        )

    reranked_results = rerank_results(
        test_query,
        relevant_results,
        top_k=3,
    )

    print("\nAfter reranking:")

    for position, result in enumerate(
        reranked_results,
        start=1,
    ):
        print(
            f"{position}. {result['file_name']} | "
            f"chunk {result['chunk_index']} | "
            f"reranker score "
            f"{result['reranker_score']:.4f}"
        )
from src.retrieval.relevance import filter_relevant_results


def test_filters_results_above_default_threshold():
    results = [
        {"id": "a", "distance": 0.10},
        {"id": "b", "distance": 0.20},
        {"id": "c", "distance": 0.21},
    ]

    filtered = filter_relevant_results(results)

    assert [result["id"] for result in filtered] == ["a", "b"]


def test_accepts_custom_threshold():
    results = [
        {"id": "a", "distance": 0.10},
        {"id": "b", "distance": 0.15},
    ]

    filtered = filter_relevant_results(
        results,
        max_distance=0.12,
    )

    assert [result["id"] for result in filtered] == ["a"]


def test_returns_empty_list_when_nothing_is_relevant():
    results = [
        {"id": "a", "distance": 0.30},
        {"id": "b", "distance": 0.40},
    ]

    assert filter_relevant_results(results) == []
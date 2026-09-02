DEFAULT_MAX_DISTANCE = 0.20


def filter_relevant_results(
    search_results,
    max_distance=DEFAULT_MAX_DISTANCE,
):
    """Remove search results above the distance threshold."""
    return [
        result
        for result in search_results
        if result["distance"] <= max_distance
    ]
def build_retrieval_query(question, history):
    """Add recent conversation context to a follow-up question."""
    if not history:
        return question.strip()

    recent_history = history[-4:]

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in recent_history
    )

    return (
        f"Previous conversation:\n"
        f"{history_text}\n"
        f"Current question: {question.strip()}"
    )


def assign_source_numbers(search_results):
    """Assign one source number to each unique file and page."""
    source_numbers = {}

    for result in search_results:
        source_key = (
            result["file_name"],
            result["page_number"],
        )

        if source_key not in source_numbers:
            source_numbers[source_key] = (
                len(source_numbers) + 1
            )

    return source_numbers


def build_context(search_results, source_numbers):
    """Format retrieved chunks as model context."""
    context_sections = []

    for result in search_results:
        source_key = (
            result["file_name"],
            result["page_number"],
        )

        source_number = source_numbers[source_key]

        source_label = (
            f"Source {source_number}: "
            f"{result['file_name']}, "
            f"page {result['page_number']}"
        )

        context_sections.append(
            f"[{source_label}]\n{result['text']}"
        )

    return "\n\n".join(context_sections)


def collect_sources(source_numbers):
    """Create the source list returned with the answer."""
    sources = []

    for source_key, source_number in source_numbers.items():
        file_name, page_number = source_key

        sources.append(
            {
                "source_number": source_number,
                "file_name": file_name,
                "page_number": page_number,
            }
        )

    return sources
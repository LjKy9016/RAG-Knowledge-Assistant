import os

from dotenv import load_dotenv
from groq import Groq

from src.retrieval.vector_store import search_chunks

# python -m src.generation.answer_generator

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_groq_client():
    """Create a Groq client using the API key from .env."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. "
            "Add it to the .env file."
        )

    return Groq(api_key=api_key)


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
    """Format retrieved chunks as context for the language model."""
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
    """Create the source list shown with the final answer."""
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


def generate_answer(question, top_k=3):
    """Retrieve relevant chunks and generate a grounded answer."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    search_results = search_chunks(
        question,
        top_k=top_k,
    )

    if not search_results:
        return {
            "answer": (
                "I could not find relevant information "
                "in the knowledge base."
            ),
            "sources": [],
        }

    source_numbers = assign_source_numbers(search_results)

    context = build_context(
        search_results,
        source_numbers,
    )

    client = get_groq_client()

    model_name = os.getenv(
        "GROQ_MODEL",
        DEFAULT_MODEL,
    )

    system_prompt = """
You are an internal knowledge assistant for Northstar Solutions.

Answer the user's question using only the supplied document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent missing information.
3. If the context does not contain the answer, clearly say that the
   information was not found in the knowledge base.
4. Answer in the same language as the user's question.
5. Keep the answer clear and concise.
6. Cite the source number after each factual answer, for example
   [Source 1].
7. Use only the source numbers provided in the document context.
""".strip()

    user_prompt = f"""
Question:
{question.strip()}

Document context:
{context}
""".strip()

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_completion_tokens=800,
    )

    answer = completion.choices[0].message.content

    return {
        "answer": answer,
        "sources": collect_sources(source_numbers),
    }


if __name__ == "__main__":
    test_question = "伦敦酒店每晚的报销限额是多少？"

    result = generate_answer(test_question)

    print(f"\nQuestion: {test_question}")
    print(f"\nAnswer:\n{result['answer']}")

    print("\nRetrieved sources:")

    for source in result["sources"]:
        print(
            f"- Source {source['source_number']}: "
            f"{source['file_name']}, "
            f"page {source['page_number']}"
        )
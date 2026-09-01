import os

from dotenv import load_dotenv
from groq import Groq

from src.generation.context_builder import (
    assign_source_numbers,
    build_context,
    build_retrieval_query,
    collect_sources,
)
from src.generation.session_store import (
    add_message,
    get_history,
    get_or_create_session,
)
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


def generate_answer(question, session_id=None, top_k=3):
    """Generate a grounded answer with conversation history."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    session_id = get_or_create_session(session_id)
    history = get_history(session_id)

    retrieval_query = build_retrieval_query(
        question,
        history,
    )

    search_results = search_chunks(
        retrieval_query,
        top_k=top_k,
    )

    if not search_results:
        refusal = (
            "I could not find relevant information "
            "in the knowledge base."
        )

        add_message(session_id, "user", question)
        add_message(session_id, "assistant", refusal)

        return {
            "session_id": session_id,
            "answer": refusal,
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
7. Use only the source numbers provided in the current document
   context.
8. Use the conversation history only to understand follow-up
   questions. Do not treat previous answers as document evidence.
""".strip()

    user_prompt = f"""
Current question:
{question.strip()}

Current document context:
{context}
""".strip()

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=800,
    )

    answer = completion.choices[0].message.content

    if not answer:
        answer = (
            "The answer could not be generated. "
            "Please try again."
        )

    add_message(session_id, "user", question)
    add_message(session_id, "assistant", answer)

    return {
        "session_id": session_id,
        "answer": answer,
        "sources": collect_sources(source_numbers),
    }


if __name__ == "__main__":
    first_question = "伦敦酒店每晚的报销限额是多少？"

    first_result = generate_answer(first_question)

    print(f"\nQuestion 1: {first_question}")
    print(f"Answer 1: {first_result['answer']}")
    print(f"Session ID: {first_result['session_id']}")

    second_question = "那英国其他地区呢？"

    second_result = generate_answer(
        second_question,
        session_id=first_result["session_id"],
    )

    print(f"\nQuestion 2: {second_question}")
    print(f"Answer 2: {second_result['answer']}")
    print(f"Session ID: {second_result['session_id']}")

    print("\nSources for the second answer:")

    for source in second_result["sources"]:
        print(
            f"- Source {source['source_number']}: "
            f"{source['file_name']}, "
            f"page {source['page_number']}"
        )
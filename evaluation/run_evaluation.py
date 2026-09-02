"""Run the RAG evaluation dataset and save raw results."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.generation.answer_generator import generate_answer
from src.generation.session_store import delete_session
from src.retrieval.retrieval_pipeline import retrieve_relevant_chunks

# python -m evaluation.run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "evaluation_set.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def load_evaluation_set():
    """Load evaluation cases from JSON."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def warm_up_models():
    """Load the embedding and reranker models before measuring latency."""
    print("Warming up embedding and reranker models...")

    retrieve_relevant_chunks(
        "What is the API rate limit?",
        top_k=2,
        use_reranker=True,
    )

    print("Model warm-up completed.\n")


def create_result_record(case, response, measured_latency_ms):
    """Convert one generated response into an evaluation record."""
    metadata = response.get("_metadata", {})

    return {
        "id": case["id"],
        "question": case["question"],
        "language": case.get("language"),
        "answerable": case.get("answerable", True),
        "reference_answer": case.get("reference_answer"),
        "expected_file": case.get("expected_file"),
        "answer": response.get("answer"),
        "sources": response.get("sources", []),
        "session_id": (
        str(response.get("session_id"))
        if response.get("session_id")
        else None
        ),
        "latency_ms": round(measured_latency_ms, 2),
        "metadata": metadata,
        "error": None,
    }


def create_error_record(case, error):
    """Create a result record when one evaluation case fails."""
    return {
        "id": case.get("id"),
        "question": case.get("question"),
        "language": case.get("language"),
        "answerable": case.get("answerable", True),
        "reference_answer": case.get("reference_answer"),
        "expected_file": case.get("expected_file"),
        "answer": None,
        "sources": [],
        "session_id": None,
        "latency_ms": None,
        "metadata": {},
        "error": str(error),
    }


def run_single_turn_case(case):
    """Run one independent question."""
    started_at = time.perf_counter()

    try:
        response = generate_answer(
            question=case["question"],
            session_id=None,
        )

        latency_ms = (time.perf_counter() - started_at) * 1000
        record = create_result_record(case, response, latency_ms)

        session_id = response.get("session_id")
        if session_id:
            delete_session(session_id)

        return record

    except Exception as error:
        return create_error_record(case, error)


def run_multi_turn_case(case):
    """Run all turns in one shared conversation session."""
    session_id = None
    turn_results = []

    for turn_index, turn in enumerate(case["turns"], start=1):
        turn_case = dict(turn)
        turn_case.setdefault(
            "id",
            f"{case['id']}_T{turn_index}",
        )

        started_at = time.perf_counter()

        try:
            response = generate_answer(
                question=turn["question"],
                session_id=session_id,
            )

            latency_ms = (time.perf_counter() - started_at) * 1000
            session_id = response.get("session_id")

            turn_record = create_result_record(
                turn_case,
                response,
                latency_ms,
            )
            turn_record["conversation_id"] = case["id"]
            turn_results.append(turn_record)

        except Exception as error:
            error_record = create_error_record(turn_case, error)
            error_record["conversation_id"] = case["id"]
            turn_results.append(error_record)

    if session_id:
        delete_session(session_id)

    return turn_results


def save_results(results):
    """Save raw evaluation results to a timestamped JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"evaluation_results_{timestamp}.json"

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(results),
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2, default=str)

    return output_path


def get_cases(dataset, primary_key, alternative_key):
    """Support either of the common evaluation-set key names."""
    return dataset.get(primary_key, dataset.get(alternative_key, []))


def main():
    dataset = load_evaluation_set()

    single_turn_cases = get_cases(
        dataset,
        "single_turn_cases",
        "single_turn",
    )
    multi_turn_cases = get_cases(
        dataset,
        "multi_turn_cases",
        "multi_turn",
    )

    warm_up_models()

    results = []

    for index, case in enumerate(single_turn_cases, start=1):
        print(
            f"[Single {index}/{len(single_turn_cases)}] "
            f"{case['question']}"
        )
        results.append(run_single_turn_case(case))

    for index, case in enumerate(multi_turn_cases, start=1):
        print(
            f"[Multi {index}/{len(multi_turn_cases)}] "
            f"{case['id']}"
        )
        results.extend(run_multi_turn_case(case))

    output_path = save_results(results)
    failed_cases = sum(result["error"] is not None for result in results)

    print("\nEvaluation run completed.")
    print(f"Total evaluated turns: {len(results)}")
    print(f"Failed turns: {failed_cases}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
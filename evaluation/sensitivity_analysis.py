"""Run selected questions with different RAG settings."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from src.generation.answer_generator import generate_answer
from src.generation.session_store import delete_session
from src.retrieval.retrieval_pipeline import retrieve_relevant_chunks

# python -m evaluation.sensitivity_analysis

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation" / "evaluation_set.json"
RESULTS_DIR = ROOT / "evaluation" / "results"

CASE_IDS = {"Q01", "Q04", "Q09", "Q12", "Q14", "Q16"}

CONFIGS = [
    {
        "name": "baseline",
        "top_k": 2,
        "use_reranker": True,
        "temperature": 0.2,
    },
    {
        "name": "precise",
        "top_k": 1,
        "use_reranker": True,
        "temperature": 0.0,
    },
    {
        "name": "faster",
        "top_k": 3,
        "use_reranker": False,
        "temperature": 0.5,
    },
]


def load_cases():
    """Load the selected single-turn cases."""
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    cases = dataset.get(
        "single_turn_cases",
        dataset.get("single_turn", []),
    )
    selected = [case for case in cases if case["id"] in CASE_IDS]

    if len(selected) != len(CASE_IDS):
        raise ValueError("Some sensitivity test cases were not found.")

    return selected


def warm_up():
    """Load retrieval models before measuring latency."""
    print("Warming up retrieval models...")
    retrieve_relevant_chunks(
        "What is the API rate limit?",
        top_k=2,
        use_reranker=True,
    )
    print("Warm-up completed.\n")


def run_case(case, config):
    """Run one evaluation case."""
    os.environ["GENERATION_TEMPERATURE"] = str(
        config["temperature"]
    )
    started_at = time.perf_counter()

    try:
        response = generate_answer(
            question=case["question"],
            session_id=None,
            top_k=config["top_k"],
            use_reranker=config["use_reranker"],
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        metadata = response.get("_metadata", {})
        session_id = response.get("session_id")

        result = {
            "config": config["name"],
            "case_id": case["id"],
            "question": case["question"],
            "reference_answer": case.get("reference_answer"),
            "expected_file": case.get("expected_file"),
            "answer": response.get("answer"),
            "sources": response.get("sources", []),
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": metadata.get("prompt_tokens", 0),
            "completion_tokens": metadata.get(
                "completion_tokens", 0
            ),
            "total_tokens": metadata.get("total_tokens", 0),
            "error": None,
        }

        if session_id:
            delete_session(session_id)

        return result

    except Exception as error:
        return {
            "config": config["name"],
            "case_id": case["id"],
            "question": case["question"],
            "expected_file": case.get("expected_file"),
            "answer": None,
            "sources": [],
            "latency_ms": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": str(error),
        }


def save_results(results):
    """Save raw sensitivity results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"sensitivity_results_{timestamp}.json"

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configs": CONFIGS,
        "case_ids": sorted(CASE_IDS),
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return output_path


def main():
    cases = load_cases()
    previous_temperature = os.environ.get(
        "GENERATION_TEMPERATURE"
    )
    results = []

    try:
        warm_up()

        for config in CONFIGS:
            print(f"Running configuration: {config['name']}")

            for case in cases:
                print(f"  [{case['id']}] {case['question']}")
                results.append(run_case(case, config))

        output_path = save_results(results)
        failed = sum(result["error"] is not None for result in results)

        print("\nSensitivity analysis completed.")
        print(f"Total runs: {len(results)}")
        print(f"Failed runs: {failed}")
        print(f"Results saved to: {output_path}")

    finally:
        if previous_temperature is None:
            os.environ.pop("GENERATION_TEMPERATURE", None)
        else:
            os.environ["GENERATION_TEMPERATURE"] = (
                previous_temperature
            )


if __name__ == "__main__":
    main()
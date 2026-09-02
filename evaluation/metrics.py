"""Calculate quality, latency, and token metrics."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

# python -m evaluation.metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
JUDGEMENTS_PATH = (
    PROJECT_ROOT / "evaluation" / "manual_judgements.json"
)

ACCURACY_TARGET = 0.80
FAITHFULNESS_TARGET = 0.85
CONTEXT_PRECISION_TARGET = 0.70
LATENCY_TARGET_MS = 10_000
LATENCY_PASS_RATE_TARGET = 0.90


def load_json(path):
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_latest_result_file():
    """Return the most recently created evaluation result."""
    result_files = list(
        RESULTS_DIR.glob("evaluation_results_*.json")
    )

    if not result_files:
        raise FileNotFoundError(
            f"No evaluation result found in {RESULTS_DIR}"
        )

    return max(result_files, key=lambda path: path.stat().st_mtime)


def calculate_percentile(values, percentile):
    """Calculate a percentile using linear interpolation."""
    if not values:
        return 0.0

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    fraction = position - lower_index

    return lower_value + (upper_value - lower_value) * fraction


def calculate_context_precision(result):
    """Measure whether returned sources match the expected file."""
    sources = result.get("sources", [])
    expected_file = result.get("expected_file")

    if not expected_file:
        return 1.0 if not sources else 0.0

    if not sources:
        return 0.0

    relevant_sources = sum(
        source.get("file_name") == expected_file
        for source in sources
    )

    return relevant_sources / len(sources)


def get_token_value(result, key):
    """Read a token value from the internal response metadata."""
    metadata = result.get("metadata", {})
    value = metadata.get(key, 0)

    if isinstance(value, (int, float)):
        return value

    return 0


def calculate_metrics(results, judgements):
    """Calculate all required evaluation metrics."""
    successful_results = [
        result for result in results
        if result.get("error") is None
    ]

    accuracy_scores = []
    faithfulness_scores = []
    context_precision_scores = []
    latencies = []

    prompt_tokens = []
    completion_tokens = []
    total_tokens = []

    for result in successful_results:
        case_id = result["id"]
        judgement = judgements.get(case_id)

        if judgement is None:
            raise KeyError(
                f"Missing manual judgement for case: {case_id}"
            )

        accuracy_scores.append(judgement["accuracy"])

        faithfulness = judgement.get("faithfulness")
        if faithfulness is not None:
            faithfulness_scores.append(faithfulness)

        context_precision_scores.append(
            calculate_context_precision(result)
        )

        latency = result.get("latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(latency)

        prompt_tokens.append(
            get_token_value(result, "prompt_tokens")
        )
        completion_tokens.append(
            get_token_value(result, "completion_tokens")
        )
        total_tokens.append(
            get_token_value(result, "total_tokens")
        )

    accuracy = mean(accuracy_scores)
    faithfulness = mean(faithfulness_scores)
    context_precision = mean(context_precision_scores)

    requests_under_target = sum(
        latency < LATENCY_TARGET_MS
        for latency in latencies
    )
    latency_pass_rate = requests_under_target / len(latencies)

    return {
        "total_cases": len(results),
        "successful_cases": len(successful_results),
        "failed_cases": len(results) - len(successful_results),
        "quality": {
            "accuracy": round(accuracy, 4),
            "accuracy_target": ACCURACY_TARGET,
            "accuracy_passed": accuracy >= ACCURACY_TARGET,
            "faithfulness": round(faithfulness, 4),
            "faithfulness_scored_cases": len(
                faithfulness_scores
            ),
            "faithfulness_target": FAITHFULNESS_TARGET,
            "faithfulness_passed": (
                faithfulness >= FAITHFULNESS_TARGET
            ),
            "context_precision": round(
                context_precision,
                4,
            ),
            "context_precision_target": (
                CONTEXT_PRECISION_TARGET
            ),
            "context_precision_passed": (
                context_precision
                >= CONTEXT_PRECISION_TARGET
            ),
        },
        "latency": {
            "average_ms": round(mean(latencies), 2),
            "p90_ms": round(
                calculate_percentile(latencies, 0.90),
                2,
            ),
            "maximum_ms": round(max(latencies), 2),
            "requests_under_10_seconds": (
                requests_under_target
            ),
            "under_10_seconds_rate": round(
                latency_pass_rate,
                4,
            ),
            "target_rate": LATENCY_PASS_RATE_TARGET,
            "target_passed": (
                latency_pass_rate
                >= LATENCY_PASS_RATE_TARGET
            ),
        },
        "tokens": {
            "average_prompt_tokens": round(
                mean(prompt_tokens),
                2,
            ),
            "average_completion_tokens": round(
                mean(completion_tokens),
                2,
            ),
            "average_total_tokens": round(
                mean(total_tokens),
                2,
            ),
            "total_tokens": sum(total_tokens),
        },
    }


def save_metrics(metrics, source_file):
    """Save calculated metrics as JSON."""
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    output_path = (
        RESULTS_DIR / f"metrics_summary_{timestamp}.json"
    )

    output = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_result_file": source_file.name,
        "metrics": metrics,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def print_metrics(metrics):
    """Print a concise metric summary."""
    quality = metrics["quality"]
    latency = metrics["latency"]
    tokens = metrics["tokens"]

    print("\nQuality metrics")
    print(f"Accuracy: {quality['accuracy']:.2%}")
    print(f"Faithfulness: {quality['faithfulness']:.2%}")
    print(
        "Context precision: "
        f"{quality['context_precision']:.2%}"
    )

    print("\nLatency metrics")
    print(f"Average latency: {latency['average_ms']} ms")
    print(f"P90 latency: {latency['p90_ms']} ms")
    print(f"Maximum latency: {latency['maximum_ms']} ms")
    print(
        "Requests under 10 seconds: "
        f"{latency['under_10_seconds_rate']:.2%}"
    )

    print("\nToken metrics")
    print(
        "Average prompt tokens: "
        f"{tokens['average_prompt_tokens']}"
    )
    print(
        "Average completion tokens: "
        f"{tokens['average_completion_tokens']}"
    )
    print(
        "Average total tokens: "
        f"{tokens['average_total_tokens']}"
    )
    print(f"Total tokens: {tokens['total_tokens']}")


def main():
    result_path = find_latest_result_file()
    result_data = load_json(result_path)
    judgement_data = load_json(JUDGEMENTS_PATH)

    metrics = calculate_metrics(
        result_data["results"],
        judgement_data["judgements"],
    )

    output_path = save_metrics(metrics, result_path)

    print(f"Result file: {result_path.name}")
    print_metrics(metrics)
    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
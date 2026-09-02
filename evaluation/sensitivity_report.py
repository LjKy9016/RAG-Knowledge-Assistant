"""Summarise the latest sensitivity analysis results."""

import json
from pathlib import Path
from statistics import mean

from evaluation.metrics import (
    calculate_context_precision,
    calculate_percentile,
)

# python -m evaluation.sensitivity_report

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation" / "results"
JUDGEMENTS_PATH = ROOT / "evaluation" / "manual_judgements.json"


def load_json(path):
    """Load one JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_latest_results():
    """Load the latest sensitivity result file."""
    files = list(RESULTS_DIR.glob("sensitivity_results_*.json"))

    if not files:
        raise FileNotFoundError(
            "No sensitivity result file was found."
        )

    latest_file = max(files, key=lambda path: path.stat().st_mtime)
    return latest_file, load_json(latest_file)


def source_precision(result):
    """Calculate source-level context precision."""
    return calculate_context_precision(
        {
            "expected_file": result.get("expected_file"),
            "sources": result.get("sources", []),
        }
    )


def summarise(config, all_results, judgements):
    """Calculate metrics for one configuration."""
    results = [
        result
        for result in all_results
        if result["config"] == config["name"]
        and result["error"] is None
    ]

    accuracy = [
        judgements[
            f"{config['name']}:{result['case_id']}"
        ]
        for result in results
    ]
    latencies = [result["latency_ms"] for result in results]
    tokens = [result["total_tokens"] for result in results]
    precision = [source_precision(result) for result in results]

    return {
        "name": config["name"],
        "top_k": config["top_k"],
        "reranker": config["use_reranker"],
        "temperature": config["temperature"],
        "accuracy": round(mean(accuracy), 4),
        "average_latency_ms": round(mean(latencies), 2),
        "p90_latency_ms": round(
            calculate_percentile(latencies, 0.90),
            2,
        ),
        "context_precision": round(mean(precision), 4),
        "average_total_tokens": round(mean(tokens), 2),
    }


def print_summary(summary):
    """Print one configuration summary."""
    print(f"\nConfiguration: {summary['name']}")
    print(f"  top_k: {summary['top_k']}")
    print(f"  reranker: {summary['reranker']}")
    print(f"  temperature: {summary['temperature']}")
    print(f"  accuracy: {summary['accuracy']:.2%}")
    print(
        f"  context precision: "
        f"{summary['context_precision']:.2%}"
    )
    print(
        f"  average latency: "
        f"{summary['average_latency_ms']} ms"
    )
    print(f"  P90 latency: {summary['p90_latency_ms']} ms")
    print(
        f"  average total tokens: "
        f"{summary['average_total_tokens']}"
    )


def main():
    result_file, data = load_latest_results()
    judgement_data = load_json(JUDGEMENTS_PATH)
    judgements = judgement_data["sensitivity_judgements"]

    summaries = [
        summarise(config, data["results"], judgements)
        for config in data["configs"]
    ]

    print(f"Result file: {result_file.name}")
    print("\nSensitivity analysis summary")

    for summary in summaries:
        print_summary(summary)


if __name__ == "__main__":
    main()
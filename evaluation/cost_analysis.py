"""Estimate LLM cost per 1,000 RAG requests."""

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation" / "results"

# Groq GPT-OSS 120B prices per one million tokens.
INPUT_PRICE = 0.15
CACHED_INPUT_PRICE = 0.075
OUTPUT_PRICE = 0.60
REQUEST_COUNT = 1000


def load_latest(pattern):
    """Load the latest result file matching a pattern."""
    files = list(RESULTS_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No result file matches: {pattern}"
        )

    latest = max(files, key=lambda path: path.stat().st_mtime)

    with latest.open("r", encoding="utf-8") as file:
        return latest, json.load(file)


def calculate_cost(name, prompt_tokens, completion_tokens):
    """Calculate average token use and cost per 1,000 calls."""
    average_prompt = mean(prompt_tokens)
    average_completion = mean(completion_tokens)

    input_cost = (
        average_prompt * REQUEST_COUNT / 1_000_000
    ) * INPUT_PRICE
    cached_input_cost = (
        average_prompt * REQUEST_COUNT / 1_000_000
    ) * CACHED_INPUT_PRICE
    output_cost = (
        average_completion * REQUEST_COUNT / 1_000_000
    ) * OUTPUT_PRICE

    return {
        "name": name,
        "average_prompt_tokens": round(average_prompt, 2),
        "average_completion_tokens": round(
            average_completion,
            2,
        ),
        "average_total_tokens": round(
            average_prompt + average_completion,
            2,
        ),
        "input_cost_per_1000_usd": round(input_cost, 4),
        "output_cost_per_1000_usd": round(
            output_cost,
            4,
        ),
        "total_cost_per_1000_usd": round(
            input_cost + output_cost,
            4,
        ),
        "cached_cost_per_1000_usd": round(
            cached_input_cost + output_cost,
            4,
        ),
    }


def baseline_cost(result_data):
    """Calculate cost from the full baseline evaluation."""
    results = [
        result
        for result in result_data["results"]
        if result.get("error") is None
    ]

    prompt_tokens = [
        result.get("metadata", {}).get(
            "prompt_tokens",
            0,
        )
        for result in results
    ]
    completion_tokens = [
        result.get("metadata", {}).get(
            "completion_tokens",
            0,
        )
        for result in results
    ]

    return calculate_cost(
        "full_baseline",
        prompt_tokens,
        completion_tokens,
    )


def sensitivity_costs(sensitivity_data):
    """Calculate cost for each sensitivity configuration."""
    costs = []

    for config in sensitivity_data["configs"]:
        results = [
            result
            for result in sensitivity_data["results"]
            if result["config"] == config["name"]
            and result.get("error") is None
        ]

        prompt_tokens = [
            result["prompt_tokens"] for result in results
        ]
        completion_tokens = [
            result["completion_tokens"] for result in results
        ]

        costs.append(
            calculate_cost(
                config["name"],
                prompt_tokens,
                completion_tokens,
            )
        )

    return costs


def save_summary(costs, baseline_file, sensitivity_file):
    """Save the cost summary."""
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    output_path = (
        RESULTS_DIR / f"cost_summary_{timestamp}.json"
    )

    output = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "currency": "USD",
        "requests": REQUEST_COUNT,
        "prices_per_million_tokens": {
            "input": INPUT_PRICE,
            "cached_input": CACHED_INPUT_PRICE,
            "output": OUTPUT_PRICE,
        },
        "baseline_result_file": baseline_file.name,
        "sensitivity_result_file": sensitivity_file.name,
        "estimates": costs,
        "note": (
            "Embedding and reranker infrastructure costs "
            "are not included."
        ),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    return output_path


def print_cost(cost):
    """Print one cost estimate."""
    print(f"\nConfiguration: {cost['name']}")
    print(
        "  average prompt tokens: "
        f"{cost['average_prompt_tokens']}"
    )
    print(
        "  average completion tokens: "
        f"{cost['average_completion_tokens']}"
    )
    print(
        "  cost per 1,000 calls: $"
        f"{cost['total_cost_per_1000_usd']:.4f}"
    )
    print(
        "  cached cost per 1,000 calls: $"
        f"{cost['cached_cost_per_1000_usd']:.4f}"
    )


def main():
    baseline_file, baseline_data = load_latest(
        "evaluation_results_*.json"
    )
    sensitivity_file, sensitivity_data = load_latest(
        "sensitivity_results_*.json"
    )

    costs = [baseline_cost(baseline_data)]
    costs.extend(sensitivity_costs(sensitivity_data))

    output_path = save_summary(
        costs,
        baseline_file,
        sensitivity_file,
    )

    print("Cost estimate for Groq GPT-OSS 120B")

    for cost in costs:
        print_cost(cost)

    print(f"\nCost summary saved to: {output_path}")


if __name__ == "__main__":
    main()
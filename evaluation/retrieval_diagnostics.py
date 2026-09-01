from src.retrieval.vector_store import search_chunks

# python -m evaluation.retrieval_diagnostics

TEST_CASES = [
    {
        "question": "How many days of annual leave do employees receive?",
        "expected_file": "01_Employee_Handbook_EN.pdf",
    },
    {
        "question": "伦敦酒店每晚的报销限额是多少？",
        "expected_file": (
            "07_Expense_Claim_Procedure_Bilingual_scanned.pdf"
        ),
    },
    {
        "question": "What is the API rate limit?",
        "expected_file": (
            "05_Orbit_Support_API_Technical_Specification_EN.pdf"
        ),
    },
    {
        "question": "临时生产权限最长有效多长时间？",
        "expected_file": "04_信息安全与隐私指南_CN.pdf",
    },
    {
        "question": "What is the capital of France?",
        "expected_file": None,
    },
    {
        "question": "What will the weather be tomorrow?",
        "expected_file": None,
    },
    {
        "question": "How do I make chocolate cake?",
        "expected_file": None,
    },
]


def run_diagnostics():
    """Compare retrieval scores for relevant and irrelevant questions."""
    print("Starting retrieval diagnostics...")
    for test_case in TEST_CASES:
        results = search_chunks(
            test_case["question"],
            top_k=3,
        )

        print(f"\nQuestion: {test_case['question']}")
        print(f"Expected file: {test_case['expected_file']}")

        for position, result in enumerate(results, start=1):
            print(
                f"  {position}. "
                f"{result['file_name']} | "
                f"chunk {result['chunk_index']} | "
                f"distance {result['distance']:.4f}"
            )


if __name__ == "__main__":
    run_diagnostics()
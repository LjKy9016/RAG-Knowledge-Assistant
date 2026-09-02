import os

from dotenv import load_dotenv


load_dotenv()


def get_boolean_setting(name, default=False):
    """Read a boolean setting from environment variables."""
    default_value = "true" if default else "false"

    return os.getenv(
        name,
        default_value,
    ).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


DEFAULT_TOP_K = int(
    os.getenv("DEFAULT_TOP_K", "2")
)

RERANK_CANDIDATE_K = int(
    os.getenv("RERANK_CANDIDATE_K", "6")
)

USE_RERANKER = get_boolean_setting(
    "USE_RERANKER",
    default=True,
)
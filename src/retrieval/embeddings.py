from sentence_transformers import SentenceTransformer

# python -m src.retrieval.embeddings

MODEL_NAME = "intfloat/multilingual-e5-small"

# Load the model only when it is first needed.
embedding_model = None


def get_embedding_model():
    """Load and return the multilingual embedding model."""
    global embedding_model

    if embedding_model is None:
        print(f"Loading embedding model: {MODEL_NAME}")

        embedding_model = SentenceTransformer(MODEL_NAME)

        print(
            f"Embedding model loaded on: "
            f"{embedding_model.device}"
        )

    return embedding_model


def embed_documents(texts):
    """Convert document chunks into normalised vectors."""
    if not texts:
        return []

    model = get_embedding_model()

    # The E5 model expects the passage prefix for stored documents.
    prepared_texts = [
        f"passage: {text.strip()}"
        for text in texts
    ]

    embeddings = model.encode(
        prepared_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def embed_query(query):
    """Convert a search query into a normalised vector."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    model = get_embedding_model()

    # The E5 model expects the query prefix for search questions.
    prepared_query = f"query: {query.strip()}"

    embedding = model.encode(
        prepared_query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embedding.tolist()


if __name__ == "__main__":
    test_documents = [
        "Full-time employees receive 25 days of annual leave.",
        "全职员工每年享有25个工作日的带薪年假。",
    ]

    document_embeddings = embed_documents(test_documents)
    query_embedding = embed_query(
        "How many days of annual leave do employees receive?"
    )

    print(f"\nDocument vectors: {len(document_embeddings)}")
    print(
        f"Document vector dimensions: "
        f"{len(document_embeddings[0])}"
    )
    print(f"Query vector dimensions: {len(query_embedding)}")
    print(f"First five values: {query_embedding[:5]}")
from pathlib import Path

import chromadb

from src.ingestion.document_loader import load_all_pdfs
from src.ingestion.text_splitter import split_documents
from src.retrieval.embeddings import embed_documents, embed_query

# python -m src.retrieval.vector_store

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "northstar_knowledge_base"


def get_chroma_client():
    """Create a persistent ChromaDB client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


def get_collection():
    """Get or create the knowledge-base collection."""
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(chunks):
    """Convert chunks into vectors and store them in ChromaDB."""
    if not chunks:
        print("No chunks to store")
        return 0

    collection = get_collection()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_documents(texts)

    ids = [chunk["chunk_id"] for chunk in chunks]

    metadatas = [
        {
            "file_name": chunk["file_name"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "used_ocr": chunk["used_ocr"],
        }
        for chunk in chunks
    ]

    # Upsert adds new chunks and updates existing chunks with the same ID.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB")

    return len(chunks)


def search_chunks(query, top_k=3):
    """Find the chunks most relevant to a question."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    for index, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][index]
        distance = results["distances"][0][index]

        matches.append(
            {
                "text": document,
                "file_name": metadata["file_name"],
                "page_number": metadata["page_number"],
                "chunk_index": metadata["chunk_index"],
                "used_ocr": metadata["used_ocr"],
                "distance": distance,
            }
        )

    return matches


def build_vector_store():
    """Load PDFs, split their text and store all chunks."""
    documents = load_all_pdfs()
    chunks = split_documents(documents)

    print(f"\nTotal chunks prepared: {len(chunks)}")

    store_chunks(chunks)

    return chunks


if __name__ == "__main__":
    build_vector_store()

    test_query = "How many days of annual leave do employees receive?"
    results = search_chunks(test_query, top_k=3)

    print(f"\nTest query: {test_query}")
    print(f"Results returned: {len(results)}")

    for position, result in enumerate(results, start=1):
        preview = result["text"][:250].replace("\n", " ")

        print(
            f"\nResult {position}\n"
            f"File: {result['file_name']}\n"
            f"Page: {result['page_number']}\n"
            f"Chunk: {result['chunk_index']}\n"
            f"Distance: {result['distance']:.4f}\n"
            f"Preview: {preview}"
        )
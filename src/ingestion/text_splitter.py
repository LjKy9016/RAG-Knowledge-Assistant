import re
from pathlib import Path

from .document_loader import load_all_pdfs


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# python -m src.ingestion.text_splitter

def normalise_text(text):
    """Remove unnecessary whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def find_split_position(text, start, end):
    """Find a suitable place to end a chunk."""
    minimum_position = start + int((end - start) * 0.6)

    separators = [
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        ". ",
        "! ",
        "? ",
        "; ",
        "；",
        ", ",
        "，",
        " ",
    ]

    best_position = -1

    for separator in separators:
        position = text.rfind(
            separator,
            minimum_position,
            end,
        )

        if position != -1:
            position += len(separator)
            best_position = max(best_position, position)

    if best_position > start:
        return best_position

    return end


def split_text(
    text,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):
    """Split text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = normalise_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        proposed_end = min(start + chunk_size, len(text))
        end = proposed_end

        if proposed_end < len(text):
            end = find_split_position(
                text,
                start,
                proposed_end,
            )

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(text):
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def split_documents(
    documents,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):
    """Split loaded PDF pages while preserving their metadata."""
    all_chunks = []

    for document in documents:
        text_chunks = split_text(
            document["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        file_stem = Path(document["file_name"]).stem

        for chunk_index, chunk_text in enumerate(
            text_chunks,
            start=1,
        ):
            chunk_id = (
                f"{file_stem}"
                f"-page-{document['page_number']}"
                f"-chunk-{chunk_index}"
            )

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "file_name": document["file_name"],
                    "page_number": document["page_number"],
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "used_ocr": document["used_ocr"],
                }
            )

    return all_chunks


if __name__ == "__main__":
    documents = load_all_pdfs()
    chunks = split_documents(documents)

    print(f"\nTotal documents/pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        preview = chunk["text"][:120].replace("\n", " ")

        print(
            f"\nChunk ID: {chunk['chunk_id']}\n"
            f"File: {chunk['file_name']}\n"
            f"Page: {chunk['page_number']}\n"
            f"Chunk number: {chunk['chunk_index']}\n"
            f"Text length: {len(chunk['text'])}\n"
            f"Preview: {preview}"
        )
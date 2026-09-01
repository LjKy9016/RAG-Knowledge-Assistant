from pathlib import Path
import pymupdf


DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"


def extract_text_from_pdf(pdf_path):
    """Extract text from each page of one PDF file."""
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            pages.append(
                {
                    "file_name": pdf_path.name,
                    "page_number": page_number,
                    "text": text,
                }
            )

    return pages


def load_all_pdfs(documents_dir=DOCUMENTS_DIR):
    """Load all PDF files from the documents folder."""
    all_pages = []
    pdf_files = sorted(documents_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {documents_dir}")
        return all_pages

    for pdf_path in pdf_files:
        try:
            pages = extract_text_from_pdf(pdf_path)
            all_pages.extend(pages)
            print(f"Loaded: {pdf_path.name} ({len(pages)} page(s))")

        except Exception as error:
            print(f"Failed to load {pdf_path.name}: {error}")

    return all_pages


if __name__ == "__main__":
    documents = load_all_pdfs()

    print(f"\nTotal pages loaded: {len(documents)}")

    for document in documents:
        text_preview = document["text"][:150].replace("\n", " ")

        print(
            f"\nFile: {document['file_name']}\n"
            f"Page: {document['page_number']}\n"
            f"Text length: {len(document['text'])}\n"
            f"Preview: {text_preview}"
        )
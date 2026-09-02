import re
from pathlib import Path

import easyocr
import numpy as np
import pymupdf
import torch


DOCUMENTS_DIR = Path(__file__).resolve().parents[2] / "documents"

# The OCR model is loaded only when a scanned page is found.
ocr_reader = None


def get_ocr_reader():
    """Load OCR models and use GPU when CUDA is available."""
    global ocr_reader

    if ocr_reader is None:
        use_gpu = torch.cuda.is_available()

        if use_gpu:
            device_name = torch.cuda.get_device_name(0)
            print(f"Loading OCR model with GPU: {device_name}")
        else:
            print("Loading OCR model with CPU")

        ocr_reader = easyocr.Reader(
            ["ch_sim", "en"],
            gpu=use_gpu,
        )

    return ocr_reader


def clean_ocr_text(text):
    """Correct common OCR errors in financial documents."""

    # Correct incorrect readings of the pound symbol.
    text = re.sub(r"(?<![A-Za-z])[ECFecf](?=\d)", "£", text)
    text = re.sub(r"\[(?=\d)", "£", text)

    # Correct common date recognition errors.
    text = re.sub(r"\b[Ii]0th", "10th", text)
    text = text.replace("10thare", "10th are")

    return text


def extract_text_with_ocr(page):
    """Convert a PDF page to an image and extract text with OCR."""
    zoom = 3
    matrix = pymupdf.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    image = np.frombuffer(
        pixmap.samples,
        dtype=np.uint8,
    ).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )

    reader = get_ocr_reader()

    text_lines = reader.readtext(
        image,
        detail=0,
        paragraph=True,
    )

    text = "\n".join(text_lines).strip()

    return clean_ocr_text(text)


def extract_text_from_pdf(pdf_path):
    """Extract text from each page of one PDF file."""
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            used_ocr = False

            # A scanned page normally contains no selectable text.
            if not text:
                print(
                    f"Scanning with OCR: {pdf_path.name}, "
                    f"page {page_number}"
                )

                text = extract_text_with_ocr(page)
                used_ocr = True

            pages.append(
                {
                    "file_name": pdf_path.name,
                    "page_number": page_number,
                    "text": text,
                    "used_ocr": used_ocr,
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

            print(
                f"Loaded: {pdf_path.name} "
                f"({len(pages)} page(s))"
            )

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
            f"OCR used: {document['used_ocr']}\n"
            f"Text length: {len(document['text'])}\n"
            f"Preview: {text_preview}"
        )
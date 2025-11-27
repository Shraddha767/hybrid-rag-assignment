from pathlib import Path
from typing import Optional
from pypdf import PdfReader


def load_pdf(pdf_path: str) -> str:
    """
    Load a multi-page PDF and return plain text concatenated from all pages.
    Assumes PDF is text-based (not pure scanned images).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    reader = PdfReader(str(path))
    pages_text = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages_text.append(text)

    full_text = "\n\n".join(pages_text)
    return full_text.strip()

"""
PDF Loader Module
=================
Extracts text page-by-page from a PDF using PyMuPDF (fitz).

Handles:
- Scanned PDFs (falls back to raw blocks)
- Encoding errors
- Empty pages
"""

from typing import List, Dict, Any
import fitz  # PyMuPDF


def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract all pages from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts: [{"page_number": int, "text": str, "word_count": int}]
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Primary extraction
        text = page.get_text("text")

        # Fallback: extract from blocks if text is empty
        if not text.strip():
            blocks = page.get_text("blocks")
            text = " ".join([b[4] for b in blocks if isinstance(b[4], str)])

        # Clean text
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        text = " ".join(text.split())  # normalize whitespace

        pages.append({
            "page_number": page_num,
            "text": text,
            "word_count": len(text.split()),
            "char_count": len(text),
        })

    doc.close()
    return pages


def get_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a PDF.

    Returns:
        Dict with title, author, page count, etc.
    """
    doc = fitz.open(pdf_path)
    meta = doc.metadata
    page_count = len(doc)
    doc.close()

    return {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
        "page_count": page_count,
        "format": meta.get("format", ""),
    }

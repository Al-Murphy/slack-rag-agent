import os
import tempfile

from pdfminer.high_level import extract_text


def parse_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract raw text from a PDF byte buffer."""
    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(pdf_bytes)
        return extract_text(path)
    finally:
        if os.path.exists(path):
            os.remove(path)

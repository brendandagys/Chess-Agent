import io
from typing import List

from pypdf import PdfReader  # type: ignore


class PDFLoader:
    """Load a PDF from raw bytes, yielding one dict per page.

    Accepts ``bytes`` directly so it works in Lambda where the content has
    already been read from S3 (no file I/O required).
    """

    def __init__(self, data: bytes, source: str = ""):
        self.data = data
        self.source = source

    def load(self) -> List[dict]:
        reader = PdfReader(io.BytesIO(self.data))
        docs: List[dict] = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            docs.append(
                {"page_content": text, "metadata": {"source": self.source, "page": page_num}}
            )

        return docs

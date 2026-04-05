from typing import List

from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain_core.documents import Document # type: ignore


class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        loader = PyPDFLoader(self.file_path)
        return loader.load()

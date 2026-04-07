import os

from .html_loader import HTMLLoader
from .pdf_loader import PDFLoader
from .pgn_loader import PGNLoader
from .txt_loader import TXTLoader
from .wikibooks_loader import WikibooksLoader

# Extension → Loader class (used when no filename-specific rule matches)
_LOADER_MAP = {
    ".pgn": PGNLoader,
    ".pdf": PDFLoader,
    ".txt": TXTLoader,
    ".md": TXTLoader,
    ".tsv": TXTLoader,
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
}


def get_loader(key: str):
    """Return the loader class for an S3 object key, or None.

    Filename-based rules are checked before the extension map so that
    specialised loaders take precedence over generic ones (e.g.
    ``wikibooks_*.html`` files use :class:`WikibooksLoader` rather than
    the generic :class:`HTMLLoader`).
    """
    basename = os.path.basename(key).lower()
    ext = os.path.splitext(basename)[1]

    # Filename-prefix rules
    if ext in (".html", ".htm") and basename.startswith("wikibooks_"):
        return WikibooksLoader

    return _LOADER_MAP.get(ext)

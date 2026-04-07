"""
RAG (Retrieval-Augmented Generation) module for the local-dev chess agent.

Contains:
  - FAISS vector store setup and document loading helpers
  - chess_knowledge LangChain tool
"""

import json
import logging
import os

import boto3  # type: ignore
from langchain.tools import tool  # type: ignore
from langchain_aws import BedrockEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_core.documents import Document  # type: ignore

from .loaders.pdf_loader import PDFLoader
from .loaders.pgn_loader import PGNLoader
from .loaders.wikibooks_loader import WikibooksLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & chunking constants
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_DEV = os.path.dirname(_HERE)
FAISS_INDEX_PATH = os.path.join(_LOCAL_DEV, "faiss_index")
DATA_DIR = os.path.join(_LOCAL_DEV, "data/pgn")

# Match the chunking constants used in ingestion/app.py
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

WIKIBOOKS_URLS = [
    "https://en.wikibooks.org/wiki/Chess_Opening_Theory",
    "https://en.wikibooks.org/wiki/Chess/Tactics",
    "https://en.wikibooks.org/wiki/Chess/Strategy",
    "https://en.wikibooks.org/wiki/Chess/Basic_Openings",
]

# ---------------------------------------------------------------------------
# Global singletons (same warm-invocation pattern as the Lambda handlers)
# ---------------------------------------------------------------------------
_bedrock_runtime = None
_embeddings = None
_vectorstore = None


def get_bedrock_runtime():
    global _bedrock_runtime
    if _bedrock_runtime is None:
        _bedrock_runtime = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
    return _bedrock_runtime


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = BedrockEmbeddings(
            client=get_bedrock_runtime(),
            model_id="amazon.titan-embed-text-v2:0",
        )
    return _embeddings


# ---------------------------------------------------------------------------
# Document loading + chunking (mirrors ingestion/app.py logic)
# ---------------------------------------------------------------------------

def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def _load_txt_documents() -> list[Document]:
    """Read every .txt file under data/ and return a list of LangChain Documents."""
    docs = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(DATA_DIR, filename)
        logger.info("Loading text file: %s", filepath)
        with open(filepath, encoding="utf-8") as fh:
            text = fh.read()
        for i, chunk in enumerate(_split_text(text)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": filename, "chunk_index": i},
                )
            )
    logger.info("Loaded %d chunks from %s", len(docs), DATA_DIR)
    return docs


def _load_pdf_documents() -> list[Document]:
    """Load all .pdf files under data/ and split them into chunks."""
    docs = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".pdf"):
            continue
        filepath = os.path.join(DATA_DIR, filename)
        logger.info("Loading PDF: %s", filepath)
        loader = PDFLoader(filepath)
        pages = loader.load()
        for page_doc in pages:
            for i, chunk in enumerate(_split_text(page_doc.page_content)):
                metadata = {**page_doc.metadata, "chunk_index": i}
                docs.append(Document(page_content=chunk, metadata=metadata))
    logger.info("Loaded %d chunks from PDF files", len(docs))
    return docs


def _load_pgn_documents() -> list[Document]:
    """Load all .pgn files under data/ and split each game into chunks."""
    docs = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".pgn"):
            continue
        filepath = os.path.join(DATA_DIR, filename)
        logger.info("Loading PGN: %s", filepath)
        loader = PGNLoader(filepath)
        games = loader.load()
        for game_doc in games:
            for i, chunk in enumerate(_split_text(game_doc.page_content)):
                metadata = {**game_doc.metadata, "chunk_index": i}
                docs.append(Document(page_content=chunk, metadata=metadata))
    logger.info("Loaded %d chunks from PGN files", len(docs))
    return docs


def _load_wikibooks_documents() -> list[Document]:
    """Fetch Wikibooks pages, split each section into chunks, and return Documents."""
    docs = []
    for url in WIKIBOOKS_URLS:
        logger.info("Loading Wikibooks page: %s", url)
        loader = WikibooksLoader(url)
        sections = loader.load()
        logger.info("Loaded %d sections from %s", len(sections), url)
        for section_doc in sections:
            for i, chunk in enumerate(_split_text(section_doc.page_content)):
                metadata = {**section_doc.metadata, "chunk_index": i}
                docs.append(Document(page_content=chunk, metadata=metadata))
    logger.info("Loaded %d chunks from %d Wikibooks URLs", len(docs), len(WIKIBOOKS_URLS))
    return docs


# ---------------------------------------------------------------------------
# FAISS vector store — idempotent: build once, reload on subsequent runs
# ---------------------------------------------------------------------------

def get_vectorstore() -> FAISS:
    """
    Return the FAISS vector store.

    Idempotency:
      - First run  → embeds all docs in data/, saves index to faiss_index/
      - Later runs → loads the saved index without re-embedding anything
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = get_embeddings()

    if os.path.isdir(FAISS_INDEX_PATH):
        logger.info("Loading existing FAISS index from %s", FAISS_INDEX_PATH)
        # allow_dangerous_deserialization is required for FAISS pickle loading;
        # safe here because we wrote the index ourselves in this local-dev context.
        _vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        logger.info("Building FAISS index from %s", DATA_DIR)

        docs = [
            *_load_txt_documents(),
            *_load_pdf_documents(),
            *_load_pgn_documents(),
            *_load_wikibooks_documents(),
        ]
        if not docs:
            raise ValueError(f"No files found in {DATA_DIR}")
        _vectorstore = FAISS.from_documents(docs, embeddings)
        _vectorstore.save_local(FAISS_INDEX_PATH)
        logger.info("FAISS index saved to %s", FAISS_INDEX_PATH)

    return _vectorstore


# ---------------------------------------------------------------------------
# Tool — RAG knowledge base
# ---------------------------------------------------------------------------

@tool
def chess_knowledge(query: str) -> str:
    """Search the chess knowledge base for relevant information.

    Use natural-language queries about openings, strategies, tactics, endgame
    techniques, or specific positions. Include the opening name and/or ECO code
    in your query when available — this dramatically improves retrieval quality.

    Returns text passages from annotated master games, chess books, opening
    theory, and strategy guides.

    Args:
        query: Natural language search query (e.g. "Sicilian Najdorf B90
               pawn structure plans", "rook endgame technique king activity").
    """
    results = get_vectorstore().similarity_search(query, k=5)
    if not results:
        logger.info("RAG | query=%r → no results", query)
        return json.dumps({"info": "No relevant documents found", "query": query})

    logger.info("RAG | query=%r → %d chunks retrieved", query, len(results))
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        preview = doc.page_content[:120].replace("\n", " ")
        logger.info("  [%d/%d] %s#%s — %s…", i, len(results), src, chunk_idx, preview)

    return "\n\n".join(doc.page_content for doc in results)

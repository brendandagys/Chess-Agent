"""
Local development runner for the chess LangChain ReAct agent.

Mirrors analyze/app.py but swaps Pinecone → FAISS (local) and runs as a CLI REPL.

Usage:
    # Set your LangSmith key first (or export it before running):
    #   export LANGCHAIN_API_KEY="your-langsmith-api-key"
    python agent.py

LangSmith tracing env vars are set below via os.environ.setdefault so that any
value already exported in your shell takes precedence.
"""

import json
import logging
import os

from dotenv import load_dotenv

# Load .env from local-dev/ — variables already set in the shell take precedence.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)

import boto3  # noqa: E402  (imported after env vars are set)
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from loaders.pdf_loader import PDFLoader
from loaders.pgn_loader import PGNLoader
from loaders.wikibooks_loader import WikibooksLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(_HERE, "faiss_index")
DATA_DIR = os.path.join(_HERE, "data/pgn")

# Match the chunking constants used in ingestion/app.py
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

WIKIBOOKS_URLS = [
    "https://en.wikibooks.org/wiki/Chess_Opening_Theory",
    "https://en.wikibooks.org/wiki/Chess/Tactics",
    "https://en.wikibooks.org/wiki/Chess/Strategy",
    "https://en.wikibooks.org/wiki/Chess/Basic_Openings"
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
    """Split text into overlapping chunks — identical algorithm to ingestion/app.py."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def _load_documents() -> list[Document]:
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
            *_load_documents(),
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
# Tool implementations
# ---------------------------------------------------------------------------

@tool
def chess_engine(fen: str) -> str:
    """Evaluate a chess position. Input should be a FEN string."""
    try:
        import chess_engine as _chess_engine  # optional native module — not required for local dev

        score = _chess_engine.evaluate(fen)
        return json.dumps({"fen": fen, "evaluation": score})
    except ImportError:
        return json.dumps({"error": "chess_engine native module not available in local dev", "fen": fen})


@tool
def rag_retrieval(query: str) -> str:
    """Retrieve relevant chess knowledge from the local FAISS document store. Input should be a natural language query."""
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


# ---------------------------------------------------------------------------
# Agent definition — identical tool set to analyze/app.py
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a chess analysis assistant. You help users analyze chess positions, "
    "suggest moves, and explain chess strategy."
)


def build_agent():
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={"temperature": 0},
    )
    return create_agent(llm, [chess_engine, rag_retrieval], system_prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Warm up the vector store (idempotent — embeds docs on first run only)
    get_vectorstore()

    agent_executor = build_agent()
    print("\nChess Agent ready (local dev, FAISS). Type 'quit' or Ctrl-C to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input or user_input.lower() in {"quit", "exit"}:
            break

        result = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]})
        messages = result.get("messages", [])
        output = messages[-1].content if messages else ""
        print(f"\nAgent: {output}\n")

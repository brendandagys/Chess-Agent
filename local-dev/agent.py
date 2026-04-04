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

import chess_engine as _engine

from dotenv import load_dotenv # type: ignore

# Load .env from local-dev/ — variables already set in the shell take precedence.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)

import boto3  # type: ignore  (imported after env vars are set)
from langchain.tools import tool  # type: ignore
from langchain_aws import BedrockEmbeddings, ChatBedrock  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_core.documents import Document  # type: ignore
from langchain_core.messages import HumanMessage  # type: ignore
from langgraph.prebuilt import create_react_agent  # type: ignore


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
# Tool implementations — chess engine
# ---------------------------------------------------------------------------

@tool
def get_top_moves(fen: str, n: int = 5) -> str:
    """Get the best moves for a position, ranked by engine evaluation.

    This is your PRIMARY move-analysis tool. Call it before recommending any
    move to the user. Returns up to ``n`` moves with scores in pawns from the
    side-to-move's perspective (positive = good for the side to move).

    Args:
        fen: FEN string of the position to analyze.
        n: Number of top moves to return (default 5).

    Returns:
        JSON list of ``{"move": "<UCI>", "score": <float>}`` objects, best first.
    """
    results = _engine.get_top_moves(fen, n)  # type: ignore
    return json.dumps([{"move": m.mv, "score": m.score} for m in results])


@tool
def evaluate_position(fen: str) -> str:
    """Get a static evaluation of a position in pawns from White's perspective.

    Positive values mean White is better; negative values mean Black is better.
    Use this for a quick overall assessment of who stands better. For comparing
    candidate moves against each other, prefer ``get_top_moves`` instead.

    Args:
        fen: FEN string of the position to evaluate.

    Returns:
        JSON object ``{"fen": "<FEN>", "evaluation": <float>}``.
    """
    score = _engine.evaluate_position(fen)  # type: ignore
    return json.dumps({"fen": fen, "evaluation": score})


@tool
def apply_moves(fen: str, moves: list[str]) -> str:
    """Apply one or more UCI moves to a position and return the resulting FEN.

    Use this for "what-if" analysis: apply a candidate move (or a sequence of
    moves), then call ``get_top_moves`` or ``evaluate_position`` on the
    resulting position to explore continuations.

    Moves MUST be in UCI format (e.g. ``e2e4``, ``e7e8q`` for promotion).

    Args:
        fen: FEN string of the starting position.
        moves: List of UCI move strings to apply in order.

    Returns:
        JSON object ``{"resulting_fen": "<FEN>"}``.
    """
    result_fen = _engine.apply_moves(fen, moves)  # type: ignore
    return json.dumps({"resulting_fen": result_fen})


@tool
def get_legal_moves(fen: str) -> str:
    """Get all legal moves in a position as UCI strings.

    ALWAYS call this to verify that a move you intend to recommend is actually
    legal before presenting it to the user. Also useful for enumerating
    candidate moves for tactical analysis.

    Args:
        fen: FEN string of the position.

    Returns:
        JSON list of UCI move strings (e.g. ``["e2e4", "d2d4", ...]``).
    """
    moves = _engine.get_legal_moves(fen)  # type: ignore
    return json.dumps(moves)


@tool
def is_square_attacked(fen: str, square: str, by_color: str) -> str:
    """Check whether a specific square is attacked by a given side.

    Use this to verify tactical observations about attacks, pins, weak squares,
    or king safety. For example, check if a king's square is attacked to
    confirm a check, or verify that an outpost square is not controlled by
    enemy pawns.

    Args:
        fen: FEN string of the position.
        square: Algebraic square name (e.g. ``"e4"``, ``"g7"``).
        by_color: ``"white"`` (or ``"w"``) / ``"black"`` (or ``"b"``).

    Returns:
        JSON object ``{"square": "<sq>", "by": "<color>", "attacked": <bool>}``.
    """
    attacked = _engine.is_square_attacked(fen, square, by_color)  # type: ignore
    return json.dumps({"square": square, "by": by_color, "attacked": attacked})


# ---------------------------------------------------------------------------
# Tool implementations — RAG knowledge base
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


# ---------------------------------------------------------------------------
# System prompt & context message builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert chess analyst with access to a chess engine and a knowledge \
base of annotated master games, opening theory, and strategy guides.

## Your tools

You have six tools. Use them proactively — do not guess at evaluations, legal \
moves, or opening theory when you can look them up.

### Chess engine tools
- **get_top_moves** — your primary analysis tool. Call it to get the best moves \
ranked by engine score before making any move recommendations.
- **evaluate_position** — quick overall assessment (from White's perspective). \
Prefer get_top_moves when you need to compare candidate moves.
- **apply_moves** — apply UCI moves to a position to get the resulting FEN. \
Use this for "what-if" lookahead: apply a candidate, then call get_top_moves \
on the result to see the opponent's best replies.
- **get_legal_moves** — enumerate all legal moves. ALWAYS verify that any \
move you recommend appears in this list before presenting it to the user.
- **is_square_attacked** — check if a square is attacked by a given side. \
Useful for verifying tactical claims (checks, pins, weak squares, king safety).

### Knowledge base
- **chess_knowledge** — search annotated games, opening theory, and strategy \
guides. When the opening name or ECO code is provided in the position context, \
include it in your query for much better results. For example, query \
"Sicilian Najdorf B90 middlegame plans" rather than just "middlegame plans".

## Move format conventions

- When **calling tools**, always use UCI notation (e.g. e2e4, g1f3, e7e8q).
- When **writing to the user**, use standard algebraic / PGN notation \
(e.g. e4, Nf3, e8=Q) for readability.

## Workflow

1. Read the position context (FEN, moves, opening, game phase, goal).
2. Call **get_top_moves** to understand the engine's assessment.
3. Call **chess_knowledge** with a targeted query that includes the opening \
name and/or game phase for relevant theory and examples.
4. Synthesize the engine analysis with the retrieved knowledge.
5. Before recommending any specific move, call **get_legal_moves** to verify \
it is legal.
6. If deeper analysis is needed, use **apply_moves** to explore a line, then \
call **get_top_moves** on the resulting position.

## Adapting to the goal

The user provides a **Goal** describing the kind of analysis they want. \
Adjust your tone, depth, and focus accordingly:

- **Coaching / teaching** — Address the player whose turn it is directly \
("you should consider…"). Explain threats, candidate moves, and the reasoning \
behind them at an accessible level. Highlight mistakes and suggest improvements.
- **Expert commentary** — Narrate objectively in the third person like a \
tournament broadcast commentator. Highlight critical moments, brilliancies, \
and subtle positional ideas.
- **Deep analysis** — Be thorough and engine-backed. Show concrete variations \
with evaluations. Compare multiple candidate moves.
- For any other goal, infer the appropriate style from the request and follow \
the same principle: use your tools to back up every claim with evidence.

## Output conventions

- Use PGN notation in prose (e.g. Nf3, not g1f3).
- Mention evaluation scores where relevant (e.g. "+0.4 in White's favor").
- Structure longer responses with clear sections or bullet points.
- When suggesting moves, always show at least the top 2–3 candidates with \
their evaluations so the user understands the alternatives.
"""

ALL_TOOLS = [
    get_top_moves,
    evaluate_position,
    apply_moves,
    get_legal_moves,
    is_square_attacked,
    chess_knowledge,
]


def build_context_message(
    fen: str,
    pgn_moves: str = "",
    opening_name: str = "",
    game_phase: str = "",
    goal: str = "",
) -> str:
    """Assemble the structured position-context message for the agent."""
    parts = [f"Position (FEN): {fen}"]
    if pgn_moves:
        parts.append(f"Moves played: {pgn_moves}")
    if opening_name:
        parts.append(f"Opening: {opening_name}")
    if game_phase:
        parts.append(f"Game phase: {game_phase}")
    parts.append("")  # blank line before goal
    parts.append(f"Goal: {goal or 'Analyze this position.'}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

MODEL_ID= "us.anthropic.claude-sonnet-4-6"

def build_agent():
    llm = ChatBedrock(
        model_id=MODEL_ID,
        model_kwargs={"temperature": 0},
    )
    return create_react_agent(model=llm, tools=ALL_TOOLS, prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Warm up the vector store (idempotent — embeds docs on first run only)
    get_vectorstore()

    agent = build_agent()
    print("\nChess Agent ready (local dev, FAISS). Type 'quit' or Ctrl-C to exit.")
    print("Enter a FEN to analyze, or free-form text. Fields are optional.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input or user_input.lower() in {"quit", "exit"}:
            break

        # Try to parse structured JSON input; fall back to plain text
        try:
            data = json.loads(user_input)
            message = build_context_message(
                fen=data.get("fen", ""),
                pgn_moves=data.get("pgn_moves", ""),
                opening_name=data.get("opening_name", ""),
                game_phase=data.get("game_phase", ""),
                goal=data.get("goal", ""),
            )
        except (json.JSONDecodeError, AttributeError):
            message = user_input

        result = agent.invoke({"messages": [HumanMessage(content=message)]})
        messages = result.get("messages", [])
        output = messages[-1].content if messages else ""
        print(f"\nAgent: {output}\n")

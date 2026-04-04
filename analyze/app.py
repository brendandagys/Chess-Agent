import json
import logging
import os

import boto3
from langchain.tools import tool
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from pinecone import Pinecone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global clients (reused across warm invocations)
_pinecone_index = None
_bedrock_runtime = None


def get_bedrock_runtime():
    global _bedrock_runtime
    if _bedrock_runtime is None:
        _bedrock_runtime = boto3.client("bedrock-runtime")
    return _bedrock_runtime


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    return _pinecone_index


# ---- Embedding & Vector Store Helpers ----

def embed(text):
    """Generate an embedding using Bedrock Titan Embed Text v2."""
    response = get_bedrock_runtime().invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def query_pinecone(query_text, top_k=5):
    """Query Pinecone for the most similar document chunks."""
    embedding = embed(query_text)
    index = get_pinecone_index()
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return [
        {"text": match["metadata"]["text"], "score": match["score"]}
        for match in results.get("matches", [])
    ]


# ---- Tool implementations — chess engine ----

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
    try:
        import chess_engine

        results = chess_engine.get_top_moves(fen, n)
        return json.dumps([{"move": m.mv, "score": m.score} for m in results])
    except ImportError:
        return json.dumps({"error": "Chess engine not available", "fen": fen})


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
    try:
        import chess_engine

        score = chess_engine.evaluate_position(fen)
        return json.dumps({"fen": fen, "evaluation": score})
    except ImportError:
        return json.dumps({"error": "Chess engine not available", "fen": fen})


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
    try:
        import chess_engine

        result_fen = chess_engine.apply_moves(fen, moves)
        return json.dumps({"resulting_fen": result_fen})
    except ImportError:
        return json.dumps({"error": "Chess engine not available", "fen": fen})


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
    try:
        import chess_engine

        moves = chess_engine.get_legal_moves(fen)
        return json.dumps(moves)
    except ImportError:
        return json.dumps({"error": "Chess engine not available", "fen": fen})


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
    try:
        import chess_engine

        attacked = chess_engine.is_square_attacked(fen, square, by_color)
        return json.dumps({"square": square, "by": by_color, "attacked": attacked})
    except ImportError:
        return json.dumps({"error": "Chess engine not available", "fen": fen})


# ---- Tool implementations — RAG knowledge base ----

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
    results = query_pinecone(query, top_k=5)
    if not results:
        return json.dumps({"info": "No relevant documents found", "query": query})
    return "\n\n".join(r["text"] for r in results)


# ---- System prompt & context message builder ----

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


def build_agent():
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={"temperature": 0},
    )
    return create_react_agent(model=llm, tools=ALL_TOOLS, prompt=SYSTEM_PROMPT)


# ---- Lambda Handler ----

def lambda_handler(event, context):
    logger.info("Event received: %s", json.dumps(event))

    body = json.loads(event.get("body") or "{}")
    fen = body.get("fen", "")
    pgn_moves = body.get("pgn_moves", "")
    opening_name = body.get("opening_name", "")
    game_phase = body.get("game_phase", "")
    goal = body.get("goal", "")
    query = body.get("query", "")

    if fen:
        message = build_context_message(
            fen=fen,
            pgn_moves=pgn_moves,
            opening_name=opening_name,
            game_phase=game_phase,
            goal=goal or query,
        )
    elif query:
        message = query
    else:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "No 'fen' or 'query' provided"}),
        }

    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(content=message)]})
    messages = result.get("messages", [])
    response_text = messages[-1].content if messages else ""

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"response": response_text}),
    }

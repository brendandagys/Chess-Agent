"""
LangChain tools for the chess agent.

Contains the chess engine tools backed by the Rust chess_engine native module
and the RAG knowledge-base tool backed by Pinecone.
"""

import json
import logging

import chess_engine as _engine  # type: ignore

from langchain.tools import tool  # type: ignore

from pinecone_client import query_pinecone

logger = logging.getLogger(__name__)


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
    logger.info("get_top_moves called", extra={"fen": fen, "n": n})
    try:
        results = _engine.get_top_moves(fen, n)  # type: ignore
        output = json.dumps([{"move": m.mv, "score": m.score} for m in results])
        logger.info(
            "get_top_moves succeeded",
            extra={"fen": fen, "n": n, "moves_returned": len(results)},
        )
        return output
    except Exception as e:
        logger.error(
            "get_top_moves failed", extra={"fen": fen, "n": n, "error": str(e)}
        )
        raise


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
    logger.info("evaluate_position called", extra={"fen": fen})
    try:
        score = _engine.evaluate_position(fen)  # type: ignore
        output = json.dumps({"fen": fen, "evaluation": score})
        logger.info("evaluate_position succeeded", extra={"fen": fen, "score": score})
        return output
    except Exception as e:
        logger.error("evaluate_position failed", extra={"fen": fen, "error": str(e)})
        raise


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
    logger.info("apply_moves called", extra={"fen": fen, "moves": moves})
    try:
        result_fen = _engine.apply_moves(fen, moves)  # type: ignore
        output = json.dumps({"resulting_fen": result_fen})
        logger.info(
            "apply_moves succeeded",
            extra={"fen": fen, "moves": moves, "resulting_fen": result_fen},
        )
        return output
    except Exception as e:
        logger.error(
            "apply_moves failed", extra={"fen": fen, "moves": moves, "error": str(e)}
        )
        raise


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
    logger.info("get_legal_moves called", extra={"fen": fen})
    try:
        moves = _engine.get_legal_moves(fen)  # type: ignore
        logger.info(
            "get_legal_moves succeeded", extra={"fen": fen, "move_count": len(moves)}
        )
        return json.dumps(moves)
    except Exception as e:
        logger.error("get_legal_moves failed", extra={"fen": fen, "error": str(e)})
        raise


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
    logger.info(
        "is_square_attacked called",
        extra={"fen": fen, "square": square, "by_color": by_color},
    )
    try:
        attacked = _engine.is_square_attacked(fen, square, by_color)  # type: ignore
        logger.info(
            "is_square_attacked succeeded",
            extra={
                "fen": fen,
                "square": square,
                "by_color": by_color,
                "attacked": attacked,
            },
        )
        return json.dumps({"square": square, "by": by_color, "attacked": attacked})
    except Exception as e:
        logger.error(
            "is_square_attacked failed",
            extra={"fen": fen, "square": square, "by_color": by_color, "error": str(e)},
        )
        raise


# ---------------------------------------------------------------------------
# Tool implementation — RAG knowledge base
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
    if not (results := query_pinecone(query, top_k=5)):
        logger.info("RAG | query=%r → no results", query)
        return json.dumps({"info": "No relevant documents found", "query": query})

    logger.info("RAG | query=%r → %d results retrieved", query, len(results))
    return "\n\n".join(r["text"] for r in results)


# ---------------------------------------------------------------------------
# Exported tool list
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    get_top_moves,
    evaluate_position,
    apply_moves,
    get_legal_moves,
    is_square_attacked,
    chess_knowledge,
]

"""
LangChain tools for the local-dev chess agent.

Contains the chess engine tools backed by the Rust chess_engine native module.
RAG tools live in rag.py.
"""

import json
import logging

import chess_engine as _engine  # type: ignore

from langchain.tools import tool  # type: ignore

from rag import chess_knowledge

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
    results = _engine.get_top_moves(fen, n) # type: ignore
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
    score = _engine.evaluate_position(fen) # type: ignore
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
    result_fen = _engine.apply_moves(fen, moves) # type: ignore
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
    moves = _engine.get_legal_moves(fen) # type: ignore
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
    attacked = _engine.is_square_attacked(fen, square, by_color) # type: ignore
    return json.dumps({"square": square, "by": by_color, "attacked": attacked})


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

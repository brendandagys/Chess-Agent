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

from dotenv import load_dotenv  # type: ignore

# Load .env from local-dev/ — variables already set in the shell take precedence.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)

from langchain_aws import ChatBedrock  # type: ignore
from langchain_core.messages import HumanMessage  # type: ignore
from langgraph.prebuilt import create_react_agent  # type: ignore

from rag import get_vectorstore
from tools import ALL_TOOLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
logger = logging.getLogger(__name__)

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

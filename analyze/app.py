import json
import logging

from langchain_aws import ChatBedrock  # type: ignore
from langchain_core.messages import HumanMessage  # type: ignore
from langgraph.prebuilt import create_react_agent  # type: ignore

from prompt import SYSTEM_PROMPT, build_context_message
from tools import ALL_TOOLS

logger = logging.getLogger()
logger.setLevel(logging.INFO)


MODEL_ID = "us.anthropic.claude-sonnet-4-6"


def build_agent():
    llm = ChatBedrock(
        model_id=MODEL_ID,
        model_kwargs={"temperature": 0, "max_tokens": 2048},
    )
    return create_react_agent(model=llm, tools=ALL_TOOLS, prompt=SYSTEM_PROMPT)


def lambda_handler(body, context):
    logger.info("Event received: %s", json.dumps(body))

    fen = body.get("fen", "")
    pgn_moves = body.get("pgn_moves", "")
    opening_name = body.get("opening_name", "")
    game_phase = body.get("game_phase", "")
    goal = body.get("goal", "")
    query = body.get("query", "")

    logger.info(
        "Parsed inputs: %s",
        json.dumps(
            {
                "fen": fen,
                "pgn_moves": pgn_moves,
                "opening_name": opening_name,
                "game_phase": game_phase,
                "goal": goal,
                "query": query,
            }
        ),
    )

    if fen:
        logger.info("Building context message from FEN position")
        message = build_context_message(
            fen=fen,
            pgn_moves=pgn_moves,
            opening_name=opening_name,
            game_phase=game_phase,
            goal=goal or query,
        )
    elif query:
        logger.info("Using raw query as message")
        message = query
    else:
        logger.warning("Request rejected: no 'fen' or 'query' provided")
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "No 'fen' or 'query' provided"}),
        }

    logger.info("Agent message: %s", message)

    try:
        agent = build_agent()
        result = agent.invoke({"messages": [HumanMessage(content=message)]})
        logger.info("Agent invocation complete. Result keys: %s", list(result.keys()))
    except Exception as e:
        logger.exception("Agent invocation failed: %s", str(e))
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Agent error: {e}"}),
        }

    messages = result.get("messages", [])
    logger.info("Agent returned %d messages", len(messages))
    for i, msg in enumerate(messages):
        logger.info(
            "Message[%d]: type=%s content=%.500s",
            i,
            type(msg).__name__,
            str(msg.content),
        )

    response_text = messages[-1].content if messages else ""
    logger.info("Final response length: %d chars", len(response_text))

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"response": response_text}),
    }

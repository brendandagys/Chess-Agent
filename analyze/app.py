import json
import logging

from langchain_aws import ChatBedrock  # type: ignore
from langchain_core.messages import HumanMessage  # type: ignore
from langgraph.prebuilt import create_react_agent  # type: ignore

from analyze.prompt import SYSTEM_PROMPT, build_context_message
from analyze.tools import ALL_TOOLS

logger = logging.getLogger()
logger.setLevel(logging.INFO)


MODEL_ID= "us.anthropic.claude-sonnet-4-6"

def build_agent():
    llm = ChatBedrock(
        model_id=MODEL_ID,
        model_kwargs={"temperature": 0, "max_tokens": 2048},
    )
    return create_react_agent(model=llm, tools=ALL_TOOLS, prompt=SYSTEM_PROMPT)


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

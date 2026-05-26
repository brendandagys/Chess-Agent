import json
import logging

import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore
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


def post_to_websocket(callback_url: str, connection_id: str, payload: dict):
    """Post a message back to the player's WebSocket connection.

    Raises botocore GoneException if the connection is already closed.
    """
    client = boto3.client("apigatewaymanagementapi", endpoint_url=callback_url)
    client.post_to_connection(
        ConnectionId=connection_id, Data=json.dumps(payload).encode("utf-8")
    )


def lambda_handler(body, context):
    logger.info("Event received: %s", json.dumps(body))

    fen = body.get("fen", "")
    pgn_moves = body.get("pgn_moves", "")
    opening_name = body.get("opening_name", "")
    game_phase = body.get("game_phase", "")
    goal = body.get("goal", "")
    query = body.get("query", "")
    connection_id = body.get("connection_id", "")
    callback_url = body.get("callback_url", "")
    analysis_type = body.get("analysis_type", "analysis")

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
                "connection_id": connection_id,
                "callback_url": callback_url,
                "analysis_type": analysis_type,
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
        if connection_id and callback_url:
            try:
                post_to_websocket(
                    callback_url,
                    connection_id,
                    {
                        "statusCode": 400,
                        "connectionId": connection_id,
                        "messages": [
                            {
                                "message": "No position to analyze",
                                "messageType": "error",
                            }
                        ],
                        "data": None,
                    },
                )
            except Exception as ws_err:
                logger.error("Failed to post error to WebSocket: %s", str(ws_err))
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
        if connection_id and callback_url:
            try:
                post_to_websocket(
                    callback_url,
                    connection_id,
                    {
                        "statusCode": 500,
                        "connectionId": connection_id,
                        "messages": [
                            {
                                "message": "Analysis failed. Please try again.",
                                "messageType": "error",
                            }
                        ],
                        "data": None,
                    },
                )
            except Exception as ws_err:
                logger.error("Failed to post error to WebSocket: %s", str(ws_err))
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

    # Post result back to the player's WebSocket connection
    if connection_id and callback_url:
        ws_payload = {
            "statusCode": 200,
            "connectionId": connection_id,
            "messages": [],
            "data": {
                "analysisType": analysis_type,
                "text": response_text,
            },
        }
        try:
            post_to_websocket(callback_url, connection_id, ws_payload)
            logger.info(
                "Successfully posted analysis result to WebSocket connection %s",
                connection_id,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "GoneException":
                logger.warning(
                    "Connection %s is gone (player disconnected)", connection_id
                )
            else:
                logger.error("Failed to post result to WebSocket: %s", str(e))
        except Exception as ws_err:
            logger.error("Failed to post result to WebSocket: %s", str(ws_err))

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"response": response_text}),
    }

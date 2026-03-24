import json
import logging
import os

import boto3
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
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


# ---- Tool Functions ----

def evaluate_position(fen: str) -> str:
    """Evaluate a chess position given a FEN string using the Rust engine."""
    try:
        import chess_engine
        score = chess_engine.evaluate(fen)
        return json.dumps({"fen": fen, "evaluation": score})
    except ImportError:
        return json.dumps({"error": "Chess engine not available", "fen": fen})


def rag_query(query: str) -> str:
    """Retrieve relevant document chunks for a query from the vector store."""
    results = query_pinecone(query, top_k=5)
    if not results:
        return json.dumps({"info": "No relevant documents found", "query": query})
    return "\n\n".join(r["text"] for r in results)


# ---- Agent Setup ----

AGENT_PROMPT = PromptTemplate.from_template(
    """You are a chess analysis assistant. You help users analyze chess positions,
suggest moves, and explain strategy. You have access to the following tools:

{tools}

Tool names: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

evaluate_tool = Tool(
    name="chess_engine",
    func=evaluate_position,
    description="Evaluate a chess position. Input should be a FEN string.",
)

rag_tool = Tool(
    name="rag_retrieval",
    func=rag_query,
    description="Retrieve relevant chess knowledge from the document store. Input should be a natural language query.",
)


def build_agent():
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={"temperature": 0},
    )
    agent = create_react_agent(llm=llm, tools=[evaluate_tool, rag_tool], prompt=AGENT_PROMPT)
    return AgentExecutor(agent=agent, tools=[evaluate_tool, rag_tool], handle_parsing_errors=True)


# ---- Lambda Handler ----

def lambda_handler(event, context):
    logger.info("Event received: %s", json.dumps(event))

    body = json.loads(event.get("body") or "{}")
    query = body.get("query", "")
    fen = body.get("fen")

    if fen:
        query = f"{query}\n\nFEN position: {fen}" if query else f"Analyze this position: {fen}"

    if not query:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "No query provided"}),
        }

    agent_executor = build_agent()
    result = agent_executor.invoke({"input": query})
    response_text = result.get("output", "")

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"response": response_text}),
    }

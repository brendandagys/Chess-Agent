import json
import logging
import os

import boto3  # type: ignore
from pinecone import Pinecone  # type: ignore

logger = logging.getLogger(__name__)

# Global clients (reused across warm invocations)
_bedrock_runtime = None
_pinecone_index = None


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


def get_embedding(text: str) -> list[float]:
    """Generate an embedding via Bedrock Titan Embed Text v2."""
    response = get_bedrock_runtime().invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def query_pinecone(query_text: str, top_k: int = 5) -> list[dict]:
    """Embed a query and return the top-k matching chunks from Pinecone."""
    embedding = get_embedding(query_text)
    index = get_pinecone_index()
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return [
        {"text": match["metadata"]["text"], "score": match["score"]}
        for match in results.get("matches", [])
    ]

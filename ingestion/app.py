import json
import logging
import os
import urllib.parse

import boto3
from pinecone import Pinecone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global clients (reused across warm invocations)
_s3 = None
_bedrock_runtime = None
_pinecone_index = None

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
UPSERT_BATCH_SIZE = 100


def get_s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


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


def load_from_s3(bucket, key):
    """Load a document from S3 and return its text content."""
    response = get_s3().get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8")


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


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


def upsert_to_pinecone(vectors):
    """Upsert a batch of vectors to Pinecone."""
    index = get_pinecone_index()
    index.upsert(vectors=vectors)


def lambda_handler(event, context):
    logger.info("Ingestion event: %s", json.dumps(event))

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

        logger.info("Processing s3://%s/%s", bucket, key)

        doc = load_from_s3(bucket, key)
        chunks = split_text(doc)

        logger.info("Document split into %d chunks", len(chunks))

        batch = []
        for i, chunk in enumerate(chunks):
            embedding = embed(chunk)
            vector_id = f"{key}#chunk{i}"
            batch.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {"text": chunk, "doc_id": key, "chunk_index": i},
            })

            if len(batch) >= UPSERT_BATCH_SIZE:
                upsert_to_pinecone(batch)
                batch = []

        if batch:
            upsert_to_pinecone(batch)

        logger.info("Finished ingesting %s (%d chunks)", key, len(chunks))

    return {"statusCode": 200, "body": json.dumps({"message": "Ingestion complete"})}

import json
import logging
import os
import typing
import urllib.parse

import boto3  # type: ignore
from pinecone import Pinecone  # type: ignore

from loaders import get_loader
from loaders.pdf_loader import PDFLoader

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global clients (reused across warm invocations)
_s3 = None
_bedrock_runtime = None
_pinecone_index = None

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
UPSERT_BATCH_SIZE = 100

# File extensions that should be read as raw bytes (not UTF-8 decoded)
_BINARY_EXTENSIONS = {".pdf"}


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


def load_bytes_from_s3(bucket, key):
    """Load a document from S3 and return its raw bytes."""
    response = get_s3().get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def get_embedding(text):
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


def delete_from_pinecone(key):
    """Delete all Pinecone vectors associated with the given S3 key."""
    index = get_pinecone_index()
    index.delete(filter={"doc_id": {"$eq": key}})
    logger.info("Deleted Pinecone vectors for doc_id=%s", key)


def lambda_handler(event, context):
    logger.info("Ingestion event: %s", json.dumps(event))

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        ext = os.path.splitext(key)[1]
        event_name = record.get("eventName", "")

        logger.info("Processing s3://%s/%s (event=%s)", bucket, key, event_name)

        if event_name.startswith("ObjectRemoved:"):
            delete_from_pinecone(key)
            continue

        LoaderClass = get_loader(key)
        if LoaderClass is None:
            logger.warning("No loader for extension %s — skipping %s", ext, key)
            continue

        # Handle overwrites where new version has fewer chunks than the previous
        delete_from_pinecone(key)

        raw = load_bytes_from_s3(bucket, key)

        if ext.lower() in _BINARY_EXTENSIONS:
            loader = typing.cast(typing.Type[PDFLoader], LoaderClass)(
                data=raw, source=key
            )
        else:
            loader = LoaderClass(text=raw.decode("utf-8"), source=key)

        documents = loader.load()

        # Chunk each document and embed
        chunk_count = 0
        batch = []
        for doc in documents:
            for i, chunk_text in enumerate(split_text(doc["page_content"])):
                embedding = get_embedding(chunk_text)
                vector_id = f"{key}#chunk{chunk_count}"

                batch.append(
                    {
                        "id": vector_id,
                        "values": embedding,
                        "metadata": {
                            "text": chunk_text,
                            **doc["metadata"],
                            "doc_id": key,
                            "chunk_index": i,
                        },
                    }
                )

                chunk_count += 1

                if len(batch) >= UPSERT_BATCH_SIZE:
                    upsert_to_pinecone(batch)
                    batch = []

        if batch:
            upsert_to_pinecone(batch)

        logger.info("Finished ingesting %s (%d chunks)", key, chunk_count)

    return {"statusCode": 200, "body": json.dumps({"message": "Ingestion complete"})}

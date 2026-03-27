import os
from numbers import Number

from pinecone import Pinecone

from kb_ingestion.embeddings import get_embedding_dimension, get_embedding_model


def _get_index_dimension(pc: Pinecone, index_name: str) -> int | None:
    description = pc.describe_index(index_name)

    if hasattr(description, "dimension"):
        return description.dimension

    if hasattr(description, "to_dict"):
        data = description.to_dict()
        if isinstance(data, dict):
            return data.get("dimension")

    if isinstance(description, dict):
        return description.get("dimension")

    return None


def validate_index_dimension(pc: Pinecone, index_name: str) -> None:
    expected_dimension = get_embedding_dimension()
    if expected_dimension is None:
        return

    index_dimension = _get_index_dimension(pc, index_name)
    if index_dimension is None or index_dimension == expected_dimension:
        return

    raise ValueError(
        "Embedding model and Pinecone index dimension do not match. "
        f"Model '{get_embedding_model()}' outputs {expected_dimension} dimensions, "
        f"but index '{index_name}' expects {index_dimension}. "
        "Set OPENAI_EMBEDDING_MODEL to a compatible model or recreate the index with the matching dimension."
    )


def _sanitize_metadata(metadata: dict) -> dict:
    sanitized = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, str | bool | Number):
            sanitized[key] = value
            continue

        if isinstance(value, list):
            string_items = [item for item in value if isinstance(item, str)]
            if string_items:
                sanitized[key] = string_items

    return sanitized

def upsert_vectors(chunks, embeddings):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX")
    validate_index_dimension(pc, index_name)
    index = pc.Index(index_name)

    vectors = []
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk["text"])
        vectors.append({
            "id": f"{chunk['metadata']['source']}-{i}",
            "values": vector,
            "metadata": _sanitize_metadata({**chunk["metadata"], "text": chunk["text"]})
        })

    index.upsert(vectors=vectors)

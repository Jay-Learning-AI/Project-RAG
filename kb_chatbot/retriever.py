import os
from typing import Any

from langchain_core.documents import Document
from pinecone import Pinecone

from kb_ingestion.embeddings import get_embeddings
from kb_ingestion.vector_store import validate_index_dimension


class PineconeRetriever:
    def __init__(self, pc: Pinecone, index_name: str, embeddings, top_k: int = 5):
        self._index = pc.Index(index_name)
        self._embeddings = embeddings
        self._top_k = top_k

    def invoke(self, question: str) -> list[Document]:
        query_vector = self._embeddings.embed_query(question)
        result = self._index.query(
            vector=query_vector,
            top_k=self._top_k,
            include_metadata=True,
        )

        matches = _extract_matches(result)
        documents = []
        for match in matches:
            metadata = dict(match.get("metadata") or {})
            page_content = metadata.pop("text", "")
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents


def _extract_matches(result: Any) -> list[dict]:
    if hasattr(result, "matches"):
        return [_match_to_dict(match) for match in result.matches]

    if isinstance(result, dict):
        return result.get("matches", [])

    return []


def _match_to_dict(match: Any) -> dict:
    if hasattr(match, "to_dict"):
        data = match.to_dict()
        if isinstance(data, dict):
            return data

    if isinstance(match, dict):
        return match

    return {
        "metadata": getattr(match, "metadata", {}) or {},
        "score": getattr(match, "score", None),
        "id": getattr(match, "id", None),
    }


def get_retriever():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX")
    validate_index_dimension(pc, index_name)

    embeddings = get_embeddings()
    return PineconeRetriever(pc, index_name, embeddings, top_k=5)

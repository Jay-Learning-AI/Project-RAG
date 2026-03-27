import os

from langchain_openai import OpenAIEmbeddings


EMBEDDING_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def get_embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def get_embedding_dimension(model: str | None = None) -> int | None:
    selected_model = model or get_embedding_model()
    return EMBEDDING_MODEL_DIMENSIONS.get(selected_model)


def get_embeddings():
    return OpenAIEmbeddings(model=get_embedding_model())
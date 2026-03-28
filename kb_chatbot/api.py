import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse

import boto3
from kb_config import load_settings
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from kb_chatbot.retriever import get_retriever
from kb_chatbot.rag_chain import build_rag_chain
from kb_chatbot.session_store import get_session_memory

load_settings()

app = FastAPI(title="Knowledge Base Chatbot")

REQUIRED_CHAT_ENV_VARS = [
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
]

SIGNED_IMAGE_TTL_SECONDS = 3600
MIN_IMAGE_RETRIEVAL_SCORE = float(os.getenv("MIN_IMAGE_RETRIEVAL_SCORE", "0.35"))
INSUFFICIENT_CONTEXT_MESSAGE = "the provided context does not contain enough information to answer this question"

SMALL_TALK_PATTERNS = {
    "hi",
    "hello",
    "hey",
    "hey there",
    "hi there",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how are you doing",
    "what's up",
    "whats up",
}

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_s3_client():
    region_name = os.getenv("AWS_REGION")
    return boto3.client("s3", region_name=region_name) if region_name else boto3.client("s3")


def _extract_s3_location(image_ref: str) -> tuple[str | None, str | None]:
    if not image_ref:
        return None, None

    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        parsed = urlparse(image_ref)
        host_parts = parsed.netloc.split(".")
        if not host_parts:
            return None, None

        bucket = host_parts[0]
        key = unquote(parsed.path.lstrip("/"))
        return (bucket or None), (key or None)

    bucket = os.getenv("S3_BUCKET_NAME")
    key = image_ref.lstrip("/")
    return (bucket or None), (key or None)


def _build_image_url(image_ref: str) -> str:
    bucket, key = _extract_s3_location(image_ref)
    if not bucket or not key:
        return image_ref

    try:
        return get_s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=SIGNED_IMAGE_TTL_SECONDS,
        )
    except Exception:
        return image_ref


def _unique_image_urls(image_refs: list[str]) -> list[str]:
    unique_urls = []
    seen = set()

    for image_ref in image_refs:
        access_url = _build_image_url(image_ref)
        if access_url in seen:
            continue
        seen.add(access_url)
        unique_urls.append(access_url)

    return unique_urls


def _select_relevant_source_docs(source_docs: list) -> list:
    if not source_docs:
        return []

    grouped = {}
    for doc in source_docs:
        source = doc.metadata.get("source") or ""
        entry = grouped.setdefault(source, {"docs": [], "count": 0, "score_sum": 0.0, "best_rank": 10**9})
        entry["docs"].append(doc)
        entry["count"] += 1

        score = doc.metadata.get("retrieval_score")
        if isinstance(score, (int, float)):
            entry["score_sum"] += float(score)

        rank = doc.metadata.get("retrieval_rank")
        if isinstance(rank, int):
            entry["best_rank"] = min(entry["best_rank"], rank)

    selected = max(
        grouped.values(),
        key=lambda item: (item["count"], item["score_sum"], -item["best_rank"]),
    )
    return selected["docs"]


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _is_small_talk(question: str) -> bool:
    normalized = _normalize_query(question)
    if not normalized:
        return True
    return normalized in SMALL_TALK_PATTERNS


def _answer_indicates_missing_context(answer: str) -> bool:
    return INSUFFICIENT_CONTEXT_MESSAGE in answer.lower()


def _has_strong_retrieval_signal(source_docs: list) -> bool:
    scores = [doc.metadata.get("retrieval_score") for doc in source_docs if isinstance(doc.metadata.get("retrieval_score"), (int, float))]
    if not scores:
        return False
    return max(scores) >= MIN_IMAGE_RETRIEVAL_SCORE


def _should_include_images(question: str, answer: str, source_docs: list) -> bool:
    if _is_small_talk(question):
        return False
    if not source_docs:
        return False
    if _answer_indicates_missing_context(answer):
        return False
    if not _has_strong_retrieval_signal(source_docs):
        return False
    return any(doc.metadata.get("image_urls") for doc in source_docs)


@lru_cache(maxsize=1)
def get_runtime():
    missing = [name for name in REQUIRED_CHAT_ENV_VARS if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Chat service is not configured. Missing environment variables: "
            f"{', '.join(missing)}. "
            "Add them to the runtime environment before using /chat."
        )

    retriever = get_retriever()
    rag_chain = build_rag_chain(retriever, get_session_memory)
    return retriever, rag_chain

class Query(BaseModel):
    session_id: str = "default"
    question: str


@app.get("/", response_class=FileResponse)
def root():
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat")
def chat(query: Query):
    try:
        _, rag_chain = get_runtime()
        result = rag_chain.invoke(
            {"question": query.question},
            config={"configurable": {"session_id": query.session_id}},
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat request failed: {exc}") from exc

    relevant_source_docs = _select_relevant_source_docs(result["source_docs"])

    image_urls = []
    if _should_include_images(query.question, result["answer"], relevant_source_docs):
        for doc in relevant_source_docs:
            image_urls.extend(doc.metadata.get("image_urls", []))

    return {
        "answer": result["answer"],
        "images": _unique_image_urls(image_urls)
    }


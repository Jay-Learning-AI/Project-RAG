import os
from functools import lru_cache
from pathlib import Path

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

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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

    image_urls = []
    for doc in result["source_docs"]:
        image_urls.extend(doc.metadata.get("image_urls", []))

    return {
        "answer": result["answer"],
        "images": list(set(image_urls))
    }


from functools import lru_cache

from kb_config import load_settings
from fastapi import FastAPI
from pydantic import BaseModel
from kb_chatbot.retriever import get_retriever
from kb_chatbot.rag_chain import build_rag_chain
from kb_chatbot.session_store import get_session_memory

load_settings()

app = FastAPI(title="Knowledge Base Chatbot")


@lru_cache(maxsize=1)
def get_runtime():
    retriever = get_retriever()
    rag_chain = build_rag_chain(retriever, get_session_memory)
    return retriever, rag_chain

class Query(BaseModel):
    session_id: str = "default"
    question: str


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat")
def chat(query: Query):
    _, rag_chain = get_runtime()
    result = rag_chain.invoke(
        {"question": query.question},
        config={"configurable": {"session_id": query.session_id}},
    )

    image_urls = []
    for doc in result["source_docs"]:
        image_urls.extend(doc.metadata.get("image_urls", []))

    return {
        "answer": result["answer"],
        "images": list(set(image_urls))
    }


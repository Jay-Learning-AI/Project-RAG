import os
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from kb_ingestion.embeddings import get_embeddings

def get_retriever():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    embeddings = get_embeddings()

    vectorstore = LangchainPinecone.from_existing_index(
        os.getenv("PINECONE_INDEX"),
        embeddings
    )

    return vectorstore.as_retriever(search_kwargs={"k": 5})

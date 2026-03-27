import os
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from kb_ingestion.embeddings import get_embeddings
from kb_ingestion.vector_store import validate_index_dimension

def get_retriever():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX")
    validate_index_dimension(pc, index_name)

    embeddings = get_embeddings()

    vectorstore = LangchainPinecone.from_existing_index(
        index_name,
        embeddings
    )

    return vectorstore.as_retriever(search_kwargs={"k": 5})

from pinecone import Pinecone
import os

def upsert_vectors(chunks, embeddings):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX"))

    vectors = []
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk["text"])
        vectors.append({
            "id": f"{chunk['metadata']['source']}-{i}",
            "values": vector,
            "metadata": {**chunk["metadata"], "text": chunk["text"]}
        })

    index.upsert(vectors=vectors)

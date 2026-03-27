
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(documents, image_urls, source):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []
    for doc in documents:
        for text in splitter.split_text(doc.page_content):
            chunks.append({
                "text": text,
                "metadata": {
                    "source": source,
                    "page": doc.metadata.get("page"),
                    "image_urls": image_urls
                }
            })
    return chunks

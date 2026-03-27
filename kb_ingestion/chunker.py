
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _get_chunk_image_urls(image_urls, page_number):
    if page_number is None:
        return image_urls

    page_prefix = f"/page{page_number}_"
    matched_urls = [url for url in image_urls if page_prefix in url]
    return matched_urls or image_urls

def create_chunks(documents, image_urls, source):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []
    for doc in documents:
        page_number = doc.metadata.get("page")
        chunk_image_urls = _get_chunk_image_urls(image_urls, page_number)
        for text in splitter.split_text(doc.page_content):
            chunks.append({
                "text": text,
                "metadata": {
                    "source": source,
                    "page": page_number,
                    "image_urls": chunk_image_urls
                }
            })
    return chunks

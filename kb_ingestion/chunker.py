
from pathlib import Path
from urllib.parse import unquote, urlparse

from langchain_text_splitters import RecursiveCharacterTextSplitter


def _get_chunk_image_urls(image_urls, page_number):
    if page_number is None:
        return image_urls

    page_prefix = f"/page{page_number}_"
    matched_urls = [url for url in image_urls if page_prefix in url]
    return matched_urls or image_urls


def _build_image_url_map(image_urls):
    return {
        Path(unquote(urlparse(url).path)).name: url
        for url in image_urls
    }


def _get_doc_image_urls(doc, image_urls, image_url_map):
    image_names = doc.metadata.get("image_names") or []
    if image_names:
        matched_urls = [image_url_map[name] for name in image_names if name in image_url_map]
        if matched_urls:
            return list(dict.fromkeys(matched_urls))

    return _get_chunk_image_urls(image_urls, doc.metadata.get("page"))

def create_chunks(documents, image_urls, source):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    image_url_map = _build_image_url_map(image_urls)
    chunks = []
    for doc in documents:
        page_number = doc.metadata.get("page")
        paragraph_index = doc.metadata.get("paragraph_index")
        chunk_image_urls = _get_doc_image_urls(doc, image_urls, image_url_map)
        for text in splitter.split_text(doc.page_content):
            chunks.append({
                "text": text,
                "metadata": {
                    "source": source,
                    "page": page_number,
                    "paragraph_index": paragraph_index,
                    "image_urls": chunk_image_urls
                }
            })
    return chunks

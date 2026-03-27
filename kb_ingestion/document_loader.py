
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader
)

SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".html": UnstructuredHTMLLoader,
}


def load_documents(file_path: str):
    """
    Load documents of any supported type and return
    LangChain Document objects.
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    loader_class = SUPPORTED_EXTENSIONS[extension]

    # Special handling for TextLoader
    if extension == ".txt":
        loader = loader_class(str(file_path), encoding="utf-8")
    else:
        loader = loader_class(str(file_path))

    return loader.load()


if __name__ == "__main__":
    import os

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
    files = os.listdir(DATA_DIR)

    if not files:
        print("No files found in data/documents/")
    else:
        for filename in files:
            file_path = os.path.join(DATA_DIR, filename)
            print(f"\n{'='*50}")
            print(f"Loading: {filename}")
            try:
                docs = load_documents(file_path)
                print(f"✅ Loaded {len(docs)} document chunk(s)")
                for i, doc in enumerate(docs[:2]):  # preview first 2 chunks
                    preview = doc.page_content[:300].replace("\n", " ").strip()
                    print(f"  [Chunk {i+1}] {preview}...")
            except ValueError as e:
                print(f"⚠️  Skipped: {e}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")

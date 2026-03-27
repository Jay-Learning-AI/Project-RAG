import os
from kb_config import load_settings

load_settings()

from kb_ingestion.document_loader import load_documents
from kb_ingestion.image_extractor import extract_images
from kb_ingestion.s3_uploader import upload_images
from kb_ingestion.chunker import create_chunks
from kb_ingestion.embeddings import get_embeddings
from kb_ingestion.vector_store import upsert_vectors

DATA_DIR = "data/documents"

REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
    "AWS_REGION",
    "S3_BUCKET_NAME",
]

def validate_env():
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"❌ Missing required environment variables: {', '.join(missing)}\n"
            "   Provide them through environment variables or GitHub Actions secrets."
        )
    print("✅ All environment variables are set.")

def main():
    validate_env()

    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    if not files:
        print("⚠️  No files found in data/documents/")
        return

    print(f"\n📂 Found {len(files)} file(s) to process.\n")

    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        print(f"{'='*50}")
        print(f"🔄 Processing: {file}")

        try:
            # 1. Load text
            documents = load_documents(file_path)
            print(f"   📄 Loaded {len(documents)} document chunk(s)")

            # 2. Extract images
            images = extract_images(file_path)
            print(f"   🖼️  Extracted {len(images)} image(s)")

            # 3. Upload images to S3
            image_urls = upload_images(images, prefix=file)
            print(f"   ☁️  Uploaded {len(image_urls)} image(s) to S3")

            # 4. Chunk text & attach image URLs
            chunks = create_chunks(documents, image_urls, source=file)
            print(f"   ✂️  Created {len(chunks)} text chunk(s)")

            # 5. Generate embeddings
            embeddings = get_embeddings()

            # 6. Store in Pinecone
            upsert_vectors(chunks, embeddings)
            print(f"   📌 Upserted vectors to Pinecone")

            print(f"   ✅ Completed ingestion for: {file}")

        except ValueError as e:
            print(f"   ⚠️  Skipped {file}: {e}")
        except Exception as e:
            print(f"   ❌ Failed to process {file}: {e}")

    print(f"\n🎉 Ingestion pipeline finished.")

if __name__ == "__main__":
    main()

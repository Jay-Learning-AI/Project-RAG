# RAG Chatbot V2

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using your own document knowledge base. Built with FastAPI, LangChain, Pinecone, and OpenAI.

## Features
- Ingests PDF, DOCX, TXT, and HTML documents
- Extracts and uploads images to S3
- Stores embeddings in Pinecone
- FastAPI endpoint for chat

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Configure secrets.

   For GitHub Actions, add these repository secrets in GitHub:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_ENV`
   - `PINECONE_INDEX`
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION`
   - `S3_BUCKET_NAME`
   - Optional: `OPENAI_EMBEDDING_MODEL` (`text-embedding-3-small` for 1536-dimension indexes, `text-embedding-3-large` for 3072-dimension indexes). If omitted or left blank, the project defaults to `text-embedding-3-small`.

   For local development, you can still place the same values in a project-root `.env` file. Environment variables provided by GitHub Secrets take precedence.
3. Run the ingestion script locally:
   ```
   python -m kb_ingestion.main
   ```
4. Start the API locally:
   ```
   uvicorn kb_chatbot.api:app --reload
   ```

## Render Deployment
1. Push this repository to GitHub.
2. In Render, create a new Blueprint or Web Service from the GitHub repository.
3. Render will detect [render.yaml](render.yaml) and create the API service automatically.
4. Add these environment variables in Render if they are not already populated from the blueprint:
   - `OPENAI_API_KEY`
   - `OPENAI_EMBEDDING_MODEL` (`text-embedding-3-small` by default)
   - `PINECONE_API_KEY`
   - `PINECONE_ENV`
   - `PINECONE_INDEX`
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION`
   - `S3_BUCKET_NAME`
5. Deploy the service. Render will start the API with:
   ```
   uvicorn kb_chatbot.api:app --host 0.0.0.0 --port $PORT
   ```
6. After deployment, verify these endpoints:
   - `/health`
   - `/docs`
   - `POST /chat`

## GitHub Actions
- The ingestion workflow at `.github/workflows/ingest.yml` reads credentials from GitHub Secrets and exposes them as environment variables for the pipeline.
- Pushing files into `data/documents/` or manually triggering the workflow runs ingestion in GitHub Actions.

## Folder Structure
- `kb_chatbot/` - API and RAG logic
- `kb_ingestion/` - Document ingestion and vector store
- `data/documents/` - Place your source documents here

## License
MIT

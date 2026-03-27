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
2. Set up your `.env` file with the following variables:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_ENV`
   - `PINECONE_INDEX`
   - `AWS_REGION`
   - `S3_BUCKET_NAME`
3. Run the ingestion script:
   ```
   python -m kb_ingestion.main
   ```
4. Start the API:
   ```
   uvicorn kb_chatbot.api:app --reload
   ```

## Folder Structure
- `kb_chatbot/` - API and RAG logic
- `kb_ingestion/` - Document ingestion and vector store
- `data/documents/` - Place your source documents here

## License
MIT

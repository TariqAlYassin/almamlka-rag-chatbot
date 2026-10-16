# Almamlaka Knowledge Assistant

A multilingual Retrieval-Augmented Generation (RAG) chatbot that answers questions using Almamlaka project documents. It retrieves relevant document passages, generates grounded answers, provides source citations, and identifies conflicting information across documents.

## Features

- **Multilingual Q&A:** Supports English and Arabic questions.
- **PDF ingestion:** Extracts text from PDF documents.
- **Text chunking:** Splits documents into overlapping chunks for retrieval.
- **Semantic search:** Uses multilingual sentence embeddings to retrieve relevant passages.
- **Vector storage:** Stores and searches document embeddings using Qdrant.
- **Grounded answers:** Uses retrieved document context to answer questions and avoids unsupported claims.
- **Conflict detection:** Performs targeted retrieval for budget and launch-date questions to help identify conflicting information.
- **Source citations:** Includes document names, page numbers, and section headings in the retrieved context and answers.
- **Automated tests:** Includes tests for ingestion, chunking, embeddings, and safety behavior.
- **Web interface:** Provides an interactive chatbot using Streamlit.

## Architecture

The application follows a RAG pipeline:

1. **Ingestion:** Read the source PDFs and extract their text and metadata.
2. **Chunking:** Split extracted text into manageable, overlapping passages.
3. **Embedding:** Convert text chunks into multilingual vector representations.
4. **Indexing:** Store vectors and associated metadata in Qdrant.
5. **Retrieval:** Search for relevant passages based on the user's question.
6. **Conflict-aware retrieval:** Retrieve additional targeted passages for questions involving budget or launch dates.
7. **Answer generation:** Use the retrieved context to generate an answer grounded in the documents.
8. **Citation and safety:** Preserve source information and decline to answer when relevant information cannot be found.

### Technology Stack

- Python
- Streamlit
- PyMuPDF
- LangChain text splitters
- Sentence Transformers
- `paraphrase-multilingual-MiniLM-L12-v2`
- Qdrant
- OpenAI API
- uv
- pytest
- Docker

## Project Structure

```text
mamlka/
├── data/
│   ├── pdfs/                 # Source PDF documents
│   └── qdrant/               # Local Qdrant data, if used
├── scripts/
│   ├── index_documents.py   # Index documents
│   ├── test_retrieval.py    # Test retrieval
│   └── test_answering.py    # Test answer generation
├── src/
│   └── mamlka/
│       ├── ingestion.py
│       ├── chunking.py
│       ├── embeddings.py
│       ├── vector_store.py
│       ├── answering.py
│       ├── conflict_detection.py
│       └── observability.py
├── tests/
├── app.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
├── Dockerfile
└── compose.yml
```

## Requirements

- Python 3.12
- uv
- An OpenAI API key
- A Qdrant instance or local Qdrant storage

## Installation

Clone the repository and enter the project directory:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd mamlka
```

Install the project dependencies:

```bash
uv sync
```

Create a local environment file from the example:

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Configure the environment variables in `.env`:

```dotenv
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

Use the Qdrant URL and API key when connecting to Qdrant Cloud. For local Qdrant storage, configure the application according to its local-storage settings.

Never commit `.env` or real API keys to version control.

## Indexing Documents

Place the source PDF documents in `data/pdfs/`, then run:

```bash
uv run python scripts/index_documents.py
```

This processes the documents, generates embeddings, and indexes the resulting chunks in Qdrant.

Run the indexing process whenever the source documents change or the vector collection needs to be rebuilt.

## Running the Application

Start the Streamlit interface:

```bash
uv run streamlit run app.py
```

Open the local URL printed in the terminal, usually:

`http://localhost:8501`

Ask questions in English or Arabic. The application retrieves relevant document passages and generates an answer using the available context.

## Testing

Run the automated test suite:

```bash
uv run pytest
```

Test document retrieval:

```bash
uv run python scripts/test_retrieval.py
```

Test answer generation:

```bash
uv run python scripts/test_answering.py
```

The answer-generation script includes questions about the budget, launch schedule, Arabic queries, and a question outside the supplied documents.

## Configuration and Deployment

The application supports local development and a hosted configuration using Streamlit Community Cloud and Qdrant Cloud.

For Streamlit Community Cloud, configure the required secrets in the app's settings:

```toml
OPENAI_API_KEY = "your_openai_api_key"
QDRANT_URL = "your_qdrant_url"
QDRANT_API_KEY = "your_qdrant_api_key"
```

Do not commit actual credentials. Ensure that the Qdrant collection has been populated before testing the deployed application.

A `Dockerfile` and `compose.yml` are included for container-based deployment where the environment is configured appropriately.


## Security

- Keep API keys in environment variables or the deployment platform's secrets manager.
- Do not commit `.env` files, credentials, or private deployment secrets.
- Restrict access to hosted services and vector databases appropriately.

## License

Specify the applicable license before public distribution.

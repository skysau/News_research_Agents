# News Research Agent (FastAPI)

A FastAPI backend for news research using OpenAI and FAISS, containerized with Docker.

## Features
- Process news article URLs
- Split and embed text using OpenAI
- Store/retrieve vectors with FAISS
- Answer user questions


## Run Command
source venv/bin/activate
streamlit run streamlit_combined.py --server.port=8502
To start the FastAPI server locally:

```bash
uvicorn app.main:app --reload
```

Or with Docker:

```bash
docker build -t news-research-agent .
docker run -p 8000:8000 --env-file .env news-research-agent
```

## Usage

### Local
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Docker
1. Build the image:
   ```bash
   docker build -t news-research-agent .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file .env news-research-agent
   ```

## API Endpoints
- `POST /process_urls/` — Process and embed news article URLs
- `POST /ask/` — Ask a question based on processed articles
- `POST /process_and_ask/` — Process URLs and answer a question in a single request

## Environment
- Requires `OPENAI_API_KEY` in your environment or `.env` file

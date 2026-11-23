# Helpdesk RAG Service

## Embedding configuration

The default embedding model is `gemini-embedding-001`, configured to return 768-dimensional vectors. Set `GEMINI_API_KEY` in the root `.env` file before running ingestion. The service uses `RETRIEVAL_DOCUMENT` for indexed text and `RETRIEVAL_QUERY` for user questions.

The database initializer creates `knowledge_chunks.embedding` as `vector(768)` and converts an existing generic vector column to that dimension. If the table already contains embeddings with a different dimension, re-embed those rows before running the initializer; the conversion intentionally fails rather than silently mixing incompatible vectors.

## Gemini request latency

Query embedding and answer generation are separate Gemini requests. The Gemini clients have no client-side request timeout, so they continue waiting for Google's response; automatic retries are disabled. Google can still terminate a request with its own deadline (HTTP 504), which the API reports as an upstream error. Application logs include the duration of each Gemini request.

The default answer model is `gemini-3.5-flash-lite`, selected for low-latency helpdesk responses. Set `LLM_MODEL` in `.env` to use another supported model. Restart the service after changing `.env`; Google API deadline errors are returned as HTTP 504.
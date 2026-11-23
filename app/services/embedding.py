"""Gemini embedding service for document chunks and search queries."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from app.config import settings

if TYPE_CHECKING:
    from app.services.chunking import TextChunk

logger = logging.getLogger(__name__)

TASK_RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
DEFAULT_BATCH_SIZE = 100


@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    """Create and cache a Gemini API client using configured credentials."""
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing from .env")
    return genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _require_model_config() -> tuple[str, int]:
    if not settings.EMBEDDING_MODEL:
        raise ValueError("EMBEDDING_MODEL is missing from .env")
    if settings.EMBEDDING_DIMENSIONS <= 0:
        raise ValueError("EMBEDDING_DIMENSIONS must be greater than 0")
    return settings.EMBEDDING_MODEL, settings.EMBEDDING_DIMENSIONS

# each embedding vector should have unit length 1.0 (l2 norm)
def _l2_normalize(values: Sequence[float]) -> list[float]:
    """Normalize vector to unit L2 length (recommended for truncated Matryoshka vectors)."""
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return [float(v) for v in values]
    return [float(v) / norm for v in values]

# send embedding request to gemini and return vectors
def _embed_batch(
    texts: list[str],
    *,
    task_type: str,
    title: str | None = None,
) -> list[list[float]]:
    """Send one embedding batch request to Gemini and return normalized vectors."""
    model, dimensions = _require_model_config()
    client = get_genai_client()

    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=dimensions,
        title=title if task_type == TASK_RETRIEVAL_DOCUMENT else None,
    )
    started_at = time.perf_counter()
    try:
        response = client.models.embed_content(
            model=model,
            contents=texts,
            config=config,
        )
    finally:
        logger.info(
            "Gemini embedding request finished in %.2fs (model=%s, texts=%d).",
            time.perf_counter() - started_at,
            model,
            len(texts),
        )

    embeddings = response.embeddings or []
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embedding(s) from '{model}', got {len(embeddings)}"
        )

    vectors: list[list[float]] = []
    for item in embeddings:
        values = item.values or []
        if len(values) != dimensions:
            raise RuntimeError(
                f"Expected {dimensions}-dim embedding from '{model}', got {len(values)}"
            )
        vectors.append(_l2_normalize(values))

    return vectors

#Generate ``RETRIEVAL_DOCUMENT`` embeddings for document chunks in batches.
def embed_documents(
    texts: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    title: str | None = None,
) -> list[list[float]]:
    """
    Generate ``RETRIEVAL_DOCUMENT`` embeddings for document chunks in batches.

    Args:
        texts: Chunk texts to embed.
        batch_size: Maximum number of chunks per Gemini API batch call.
        title: Optional document title passed to Gemini for retrieval quality.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if not texts:
        return []

    cleaned = [t.strip() for t in texts]
    if any(not t for t in cleaned):
        raise ValueError("Cannot embed empty or whitespace-only document text")

    vectors: list[list[float]] = []
    for start in range(0, len(cleaned), batch_size):
        batch = cleaned[start : start + batch_size]
        vectors.extend(
            _embed_batch(batch, task_type=TASK_RETRIEVAL_DOCUMENT, title=title)
        )

    logger.info(
        "Generated %d document embedding(s) in %d batch(es) (model=%s, dims=%d).",
        len(vectors),
        math.ceil(len(cleaned) / batch_size),
        settings.EMBEDDING_MODEL,
        settings.EMBEDDING_DIMENSIONS,
    )
    return vectors


def embed_document(text: str, *, title: str | None = None) -> list[float]:
    """Generate a single ``RETRIEVAL_DOCUMENT`` embedding vector."""
    return embed_documents([text], title=title)[0]

# takes chunks and sets them into batches
def embed_chunks(
    chunks: Sequence[TextChunk],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    title: str | None = None,
) -> list[list[float]]:
    """Generate ``RETRIEVAL_DOCUMENT`` embeddings for a sequence of ``TextChunk`` items."""
    return embed_documents(
        [chunk.content for chunk in chunks],
        batch_size=batch_size,
        title=title,
    )

# takes user query and generates embeddings by calling gemini
def embed_query(question: str) -> list[float]:
    """Generate a ``RETRIEVAL_QUERY`` embedding vector for a user search question."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("Query text cannot be empty")

    vector = _embed_batch([cleaned], task_type=TASK_RETRIEVAL_QUERY)[0]
    logger.debug(
        "Generated query embedding (model=%s, dims=%d).",
        settings.EMBEDDING_MODEL,
        len(vector),
    )
    return vector


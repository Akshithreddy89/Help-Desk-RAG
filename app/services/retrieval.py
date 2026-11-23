"""Vector retrieval service using Supabase pgvector and Gemini query embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pgvector import Vector

from app.database.connection import get_db_connection
from app.services.embedding import embed_query

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved knowledge base chunk with similarity score and document context."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str
    similarity: float
    distance: float
    heading: str | None = None
    question: str | None = None
    page_number: int | None = None
    page_numbers: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)



#Find the top 5 chunks of text in the knowledge base whose meaning is closest to the question and calculate a match score, and rank them from best to worst
def _build_search_query(*, has_distance_filter: bool) -> str:
    """
    Construct optimized vector similarity search query.

    Uses cosine distance (<=>) with HNSW index scan on knowledge_chunks.embedding.
    """
    distance_condition = (  #this condition is used to filter out the chunks that are not relevant to the query(Any chunk with a match weaker than min_similarity, will be ignore)
        "AND (c.embedding <=> %(vec)s) <= %(max_distance)s" 
        if has_distance_filter
        else ""
    )

    return f"""
        SELECT
            c.id,
            c.document_id,
            d.filename,
            c.chunk_index,
            c.content,
            c.metadata,
            (c.embedding <=> %(vec)s) AS distance,
            1.0 - (c.embedding <=> %(vec)s) AS similarity
        FROM public.knowledge_chunks c
        JOIN public.knowledge_documents d ON d.id = c.document_id
        WHERE d.status = 'completed'
          {distance_condition}
        ORDER BY c.embedding <=> %(vec)s
        LIMIT %(limit)s;
    """

def _execute_search(
    conn: Any,
    query_vector: list[float],
    *,
    top_k: int,
    min_similarity: float | None = None,
    ef_search: int | None = None,
) -> list[RetrievedChunk]:
    """Execute vector search query against a database connection."""
    has_filter = min_similarity is not None
    sql = _build_search_query(has_distance_filter=has_filter)

    params: dict[str, Any] = {
        "vec": Vector(query_vector),
        "limit": top_k,
    }
    if has_filter:
        # Cosine distance = 1.0 - Cosine similarity
        params["max_distance"] = 1.0 - min_similarity

    with conn.cursor() as cur:
        if ef_search is not None and ef_search > 0:
            # Tune HNSW search recall dynamic list size if requested
            cur.execute("SET LOCAL hnsw.ef_search = %s;", (ef_search,))

        cur.execute(sql, params)
        rows = cur.fetchall()

    chunks: list[RetrievedChunk] = []
    for row in rows:
        (
            chunk_id,
            doc_id,
            filename,
            chunk_index,
            content,
            meta,
            distance,
            similarity,
        ) = row

        metadata = dict(meta or {})
        heading = metadata.get("heading")
        question = metadata.get("question")
        page_num = metadata.get("page_number")
        page_nums = metadata.get("page_numbers") or ([page_num] if page_num is not None else [])

        chunks.append(
            RetrievedChunk(
                chunk_id=str(chunk_id),
                document_id=str(doc_id),
                filename=filename,
                chunk_index=chunk_index,
                content=content,
                similarity=round(float(similarity), 4),
                distance=round(float(distance), 4),
                heading=heading,
                question=question,
                page_number=page_num,
                page_numbers=page_nums,
                metadata=metadata,
            )
        )

    return chunks


def retrieve(   #1) this will execute when a user asks question
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float | None = None,
    conn: Any | None = None,
    ef_search: int | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k relevant knowledge chunks for a user search question.

    1. Embeds question using ``embed_query`` (Gemini RETRIEVAL_QUERY).
    2. Runs HNSW cosine similarity search (<=>) joined with ``knowledge_documents``.
    3. Filters out incomplete/failed documents and matches below ``min_similarity`` (if set).
    4. Reuses provided ``conn`` (e.g. from FastAPI Depends(get_db)) or borrows from ``db_pool``.

    Args:
        query: Natural language user question.
        top_k: Number of most similar chunks to return.
        min_similarity: Optional threshold (0.0 - 1.0). Chunks with lower similarity are excluded.
        conn: Optional active database connection.
        ef_search: Optional HNSW search candidate pool size (default pgvector is 40).

    Returns:
        Ranked list of ``RetrievedChunk`` items ordered by highest similarity.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if min_similarity is not None and not (0.0 <= min_similarity <= 1.0):
        raise ValueError("min_similarity must be between 0.0 and 1.0")

    query_vector = embed_query(cleaned_query) #2) this will execute embed_query function to get the vector of the question

    if conn is not None:
        results = _execute_search(
            conn,
            query_vector,
            top_k=top_k,
            min_similarity=min_similarity,
            ef_search=ef_search,
        )
    else:
        with get_db_connection() as managed_conn:
            results = _execute_search(
                managed_conn,
                query_vector,
                top_k=top_k,
                min_similarity=min_similarity,
                ef_search=ef_search,
            )

    logger.info(
        "Retrieved %d chunk(s) for query: %.60s (top_k=%d, min_similarity=%s)",
        len(results),
        cleaned_query,
        top_k,
        min_similarity,
    )
    return results


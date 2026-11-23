"""Document ingestion pipeline: extract, chunk, embed, and persist to pgvector."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pgvector import Vector
from psycopg.types.json import Jsonb

from app.services.chunking import chunk_document
from app.services.embedding import DEFAULT_BATCH_SIZE, embed_chunks
from app.services.extraction import extract_document, list_knowledge_base_files

logger = logging.getLogger(__name__)

# for compare checksum and status and if same then skip
_SELECT_DOCUMENT_SQL = """
    SELECT id, checksum, status
    FROM public.knowledge_documents
    WHERE filename = %s
    ORDER BY created_at DESC
    LIMIT 1
"""

# insert new file name and checksum and status processing
_INSERT_DOCUMENT_SQL = """
    INSERT INTO public.knowledge_documents (filename, checksum, status)
    VALUES (%s, %s, 'processing')
    RETURNING id
"""
_UPDATE_DOCUMENT_PROCESSING_SQL = """
    UPDATE public.knowledge_documents
    SET checksum = %s, status = 'processing', updated_at = NOW()
    WHERE id = %s
"""
_UPDATE_DOCUMENT_STATUS_SQL = """
    UPDATE public.knowledge_documents
    SET status = %s, updated_at = NOW()
    WHERE id = %s
"""
_DELETE_CHUNKS_SQL = "DELETE FROM public.knowledge_chunks WHERE document_id = %s"
_INSERT_CHUNKS_SQL = """
    INSERT INTO public.knowledge_chunks (
        document_id, chunk_index, content, embedding, metadata
    )
    VALUES (%s, %s, %s, %s, %s)
"""
_COMPLETE_DOCUMENT_SQL = """
    UPDATE public.knowledge_documents
    SET status = 'completed', checksum = %s, updated_at = NOW()
    WHERE id = %s
"""


@dataclass(frozen=True)
class IngestionResult:
    """Outcome summary for ingesting a single knowledge base document."""

    filename: str
    status: str  # "completed" | "skipped" | "failed"
    document_id: str | None = None
    chunk_count: int = 0
    checksum: str = ""
    error: str | None = None


def compute_checksum(file_path: str | Path, chunk_size: int = 65536) -> str:
    """SHA-256 hex digest streamed from disk (does not load the whole file)."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _mark_document_status(conn: Any, doc_id: Any, status: str) -> None:
    with conn.cursor() as cur:
        cur.execute(_UPDATE_DOCUMENT_STATUS_SQL, (status, doc_id))
    conn.commit()


def ingest_file(
    conn: Any,
    file_path: str | Path,
    *,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IngestionResult:
    """
    Ingest one knowledge base file into ``knowledge_documents`` and ``knowledge_chunks``.

    Unchanged files whose checksum already matches a ``completed`` row are skipped
    unless ``force=True``. Chunks and embeddings are written in one transaction.
    """
    path = Path(file_path)
    filename = path.name
    if not path.is_file():
        return IngestionResult(
            filename=filename,
            status="failed",
            error=f"File not found: {path}",
        )

    checksum = compute_checksum(path)   #calculate checksum of the file

    with conn.cursor() as cur:
        cur.execute(_SELECT_DOCUMENT_SQL, (filename,)) # for compare checksum in db
        existing = cur.fetchone()

        if existing and not force:
            existing_id, existing_checksum, existing_status = existing
            if existing_checksum == checksum and existing_status == "completed": #if checksum and status is same then skip
                logger.info("Skipping unchanged file '%s'.", filename)
                return IngestionResult(
                    filename=filename,
                    status="skipped",
                    document_id=str(existing_id),
                    checksum=checksum,
                )

        if existing:
            doc_id = existing[0]
            cur.execute(_UPDATE_DOCUMENT_PROCESSING_SQL, (checksum, doc_id))
        else:
            cur.execute(_INSERT_DOCUMENT_SQL, (filename, checksum))  #insert into public.knowledge_documents
            doc_id = cur.fetchone()[0]
    conn.commit()

    try:
        chunks = chunk_document(extract_document(path)) # chunk_document function call here
        if not chunks:
            raise ValueError(f"No chunks produced for document '{filename}'")

        vectors = embed_chunks(chunks, batch_size=batch_size, title=filename) #embed_chunks function call here
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch for '{filename}': {len(vectors)} != {len(chunks)}"
            )

        rows = [
            (doc_id, chunk.chunk_index, chunk.content, Vector(vec), Jsonb(chunk.metadata))
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]

        with conn.transaction(), conn.cursor() as cur: #conn.transaction() here using transaction
            cur.execute(_DELETE_CHUNKS_SQL, (doc_id,))
            cur.executemany(_INSERT_CHUNKS_SQL, rows)
            cur.execute(_COMPLETE_DOCUMENT_SQL, (checksum, doc_id))

        logger.info("Ingested '%s' (%d chunks).", filename, len(chunks))
        return IngestionResult(
            filename=filename,
            status="completed",
            document_id=str(doc_id),
            chunk_count=len(chunks),
            checksum=checksum,
        )
    except Exception as exc:
        conn.rollback()
        try:
            _mark_document_status(conn, doc_id, "failed")
        except Exception:
            logger.exception("Failed to mark document '%s' as failed.", filename)
        logger.exception("Ingestion failed for '%s'.", filename)
        return IngestionResult(
            filename=filename,
            status="failed",
            document_id=str(doc_id),
            checksum=checksum,
            error=str(exc),
        )


def ingest_directory(
    conn: Any,
    directory: str | Path | None = None,
    *,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[IngestionResult]:
    """Ingest all supported files in ``directory`` (defaults to ``knowledge_base/documents``)."""
    files = list_knowledge_base_files(directory)
    if not files:
        logger.warning("No supported documents found to ingest.")
        return []

    return [
        ingest_file(conn, file_path, force=force, batch_size=batch_size)
        for file_path in files
    ]

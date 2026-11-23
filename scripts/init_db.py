import sys
from pathlib import Path

# Add project root to sys.path so 'app' can be imported when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import psycopg

from app.config import settings

INIT_SQL = f"""
-- 1. Ensure the pgvector extension exists in Supabase/PostgreSQL
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- Make sure the extensions schema is in search_path so 'VECTOR' type is recognized
SET search_path TO public, extensions;

-- 2. Create knowledge_documents table
CREATE TABLE IF NOT EXISTS public.knowledge_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    checksum TEXT,
    status VARCHAR(30) NOT NULL DEFAULT 'pending'
        CHECK (status IN (
            'pending',
            'processing',
            'completed',
            'failed'
        )),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Create knowledge_chunks table
CREATE TABLE IF NOT EXISTS public.knowledge_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL
        REFERENCES public.knowledge_documents(id)
        ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding extensions.vector({settings.EMBEDDING_DIMENSIONS}),
    metadata JSONB DEFAULT '{{}}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_document_chunk
        UNIQUE (document_id, chunk_index)
);

-- Constrain a previously generic vector column to the configured model dimension.
ALTER TABLE public.knowledge_chunks
    ALTER COLUMN embedding
    TYPE extensions.vector({settings.EMBEDDING_DIMENSIONS})
    USING embedding::extensions.vector({settings.EMBEDDING_DIMENSIONS});

-- 4. Create performance indexes
-- HNSW vector index for fast semantic cosine similarity search
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding_hnsw
ON public.knowledge_chunks
USING hnsw (embedding extensions.vector_cosine_ops);

-- Quickly find chunks belonging to a document
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document_id
ON public.knowledge_chunks(document_id);

-- Quickly filter documents by processing status
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_status
ON public.knowledge_documents(status);

-- Quickly find documents using their checksum
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_checksum
ON public.knowledge_documents(checksum);
"""

VERIFY_SQL = """
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('knowledge_documents', 'knowledge_chunks');
"""

VERIFY_INDEXES_SQL = """
SELECT indexname 
FROM pg_indexes 
WHERE schemaname = 'public' 
  AND tablename = 'knowledge_chunks'
ORDER BY indexname;
"""


def init_database():
    if not settings.DATABASE_URL:
        print("Error: DATABASE_URL is not set in .env")
        sys.exit(1)

    print("Connecting to PostgreSQL/Supabase database...")
    try:
        with psycopg.connect(settings.DATABASE_URL) as conn:
            with conn.cursor() as cur:
                print("Executing schema initialization...")
                cur.execute(INIT_SQL)
                conn.commit()

                print("Verifying created tables...")
                cur.execute(VERIFY_SQL)
                created_tables = [row[0] for row in cur.fetchall()]
                print(f"Verified tables in database: {created_tables}")

                print("Verifying indexes on knowledge_chunks...")
                cur.execute(VERIFY_INDEXES_SQL)
                created_indexes = [row[0] for row in cur.fetchall()]
                print(f"Verified indexes: {created_indexes}")

        print("\nDatabase initialization completed successfully!")
    except Exception as exc:
        print(f"\nFailed to initialize database: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    init_database()
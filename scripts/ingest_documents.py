"""CLI: ingest knowledge base documents into Supabase pgvector.

Drop a PDF (or .txt/.md) into knowledge_base/documents, then run:

    python scripts/ingest_documents.py

Unchanged completed files are skipped. New or edited files are extracted,
chunked, embedded, and stored. Re-embed everything with --force.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent   # check inside knowledge base folder
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database.connection import close_pool, get_db_connection, open_pool
from app.services.ingestion import IngestionResult, ingest_directory, ingest_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract, chunk, embed, and store knowledge base documents."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Optional file or folder. Default: knowledge_base/documents.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest files even if the checksum already matches a completed row.",
    )
    return parser.parse_args()


def _print_summary(results: list[IngestionResult]) -> None:
    if not results:
        print("No documents found to ingest.")
        return

    print("\nIngestion Summary:")
    print("-" * 70)
    for res in results:
        details = f"chunks={res.chunk_count}" if res.status == "completed" else ""
        if res.error:
            details = f"error={res.error}"
        print(f"  {res.filename:<32} status={res.status:<10} {details}".rstrip())
    print("-" * 70)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    target: Path | None = args.path.resolve() if args.path is not None else None
    if target is not None and not target.exists():
        print(f"Path not found: {target}", file=sys.stderr)
        return 1

    open_pool()
    try:
        with get_db_connection() as conn:
            if target is None or target.is_dir():
                results = ingest_directory(conn, directory=target, force=args.force)   
            else:
                results = [ingest_file(conn, target, force=args.force)]  # ingest service file is called
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        close_pool()

    _print_summary(results)
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

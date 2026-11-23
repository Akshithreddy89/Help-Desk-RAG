"""Tests for document ingestion service."""

from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pgvector import Vector
from psycopg.types.json import Jsonb

from app.services import ingestion


def _make_mock_conn(existing_row=None, returning_id="doc-uuid-1"):
    cursor = MagicMock()
    cursor.fetchone.side_effect = (
        [existing_row] if existing_row is not None else [None, (returning_id,)]
    )

    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    conn.cursor.return_value.__exit__.return_value = False
    conn.transaction.return_value = nullcontext()
    return conn, cursor


class IngestionTests(unittest.TestCase):
    """Verify checksum skipping, status tracking, and transactional chunk storage."""

    def test_skips_unchanged_completed_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_path = Path(tmpdir) / "faq.txt"
            doc_path.write_text("Q1: Why?\nBecause.", encoding="utf-8")
            checksum = ingestion.compute_checksum(doc_path)

            conn, cursor = _make_mock_conn(
                existing_row=("doc-uuid-1", checksum, "completed")
            )

            with (
                patch("app.services.ingestion.extract_document") as mock_extract,
                patch("app.services.ingestion.embed_chunks") as mock_embed,
            ):
                result = ingestion.ingest_file(conn, doc_path)

            self.assertEqual(result.status, "skipped")
            self.assertEqual(result.document_id, "doc-uuid-1")
            self.assertEqual(result.checksum, checksum)
            mock_extract.assert_not_called()
            mock_embed.assert_not_called()
            cursor.executemany.assert_not_called()

    def test_ingests_new_document_in_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_path = Path(tmpdir) / "faq.txt"
            doc_path.write_text(
                "Q1: How to fix Wi-Fi?\nRestart the router.\n\n"
                "Q2: Why is it slow?\nMove closer to the router.",
                encoding="utf-8",
            )

            conn, cursor = _make_mock_conn(existing_row=None, returning_id="doc-uuid-2")
            fake_vectors = [[0.1] * 768, [0.2] * 768]

            with patch("app.services.ingestion.embed_chunks", return_value=fake_vectors):
                result = ingestion.ingest_file(conn, doc_path)

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.document_id, "doc-uuid-2")
            self.assertEqual(result.chunk_count, 2)

            # Verify executemany inserted Vector and Jsonb rows
            insert_sql, rows = cursor.executemany.call_args[0]
            self.assertIn("INSERT INTO public.knowledge_chunks", insert_sql)
            self.assertEqual(len(rows), 2)
            self.assertIsInstance(rows[0][3], Vector)
            self.assertIsInstance(rows[0][4], Jsonb)

            executed_sql = " ".join(str(call.args[0]) for call in cursor.execute.call_args_list)
            self.assertIn("'processing'", executed_sql)
            self.assertIn("status = 'completed'", executed_sql)

    def test_marks_failed_when_embedding_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_path = Path(tmpdir) / "faq.txt"
            doc_path.write_text("Q1: Test?\nAnswer.", encoding="utf-8")

            conn, cursor = _make_mock_conn(existing_row=None, returning_id="doc-uuid-3")

            with patch(
                "app.services.ingestion.embed_chunks",
                side_effect=RuntimeError("Gemini API failure"),
            ):
                result = ingestion.ingest_file(conn, doc_path)

            self.assertEqual(result.status, "failed")
            self.assertIn("Gemini API failure", result.error or "")
            conn.rollback.assert_called_once()
            failed_updates = [
                call.args[1]
                for call in cursor.execute.call_args_list
                if "SET status = %s" in str(call.args[0])
            ]
            self.assertEqual(failed_updates, [("failed", "doc-uuid-3")])


if __name__ == "__main__":
    unittest.main()
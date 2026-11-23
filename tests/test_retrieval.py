"""Unit tests for vector retrieval service."""

from contextlib import nullcontext
import unittest
from unittest.mock import MagicMock, patch

from pgvector import Vector

from app.services import retrieval


class RetrievalTests(unittest.TestCase):
    """Verify vector search query building, threshold filtering, and result mapping."""

    def test_empty_query_returns_empty_list(self) -> None:
        self.assertEqual(retrieval.retrieve("   "), [])

    def test_invalid_parameters_raise(self) -> None:
        with self.assertRaises(ValueError):
            retrieval.retrieve("test", top_k=0)
        with self.assertRaises(ValueError):
            retrieval.retrieve("test", min_similarity=1.5)
        with self.assertRaises(ValueError):
            retrieval.retrieve("test", min_similarity=-0.1)

    def test_retrieve_executes_hnsw_query_with_conn(self) -> None:
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            (
                "chunk-uuid-1",
                "doc-uuid-1",
                "wifi_troubleshooting.pdf",
                0,
                "1. My Wi-Fi is connected but there is no internet...",
                {
                    "heading": "Wi-Fi Troubleshooting",
                    "question": "1. My Wi-Fi is connected but there is no internet?",
                    "page_number": 1,
                    "page_numbers": [1],
                },
                0.15234,  # distance
                0.84766,  # similarity
            )
        ]
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor

        fake_vector = [0.05] * 768
        with patch("app.services.retrieval.embed_query", return_value=fake_vector) as mock_embed:
            results = retrieval.retrieve(
                "connected but no internet",
                top_k=3,
                conn=conn,
            )

        mock_embed.assert_called_once_with("connected but no internet")
        self.assertEqual(len(results), 1)

        item = results[0]
        self.assertEqual(item.chunk_id, "chunk-uuid-1")
        self.assertEqual(item.filename, "wifi_troubleshooting.pdf")
        self.assertEqual(item.chunk_index, 0)
        self.assertEqual(item.similarity, 0.8477)
        self.assertEqual(item.distance, 0.1523)
        self.assertEqual(item.page_number, 1)
        self.assertEqual(item.heading, "Wi-Fi Troubleshooting")

        sql, params = cursor.execute.call_args[0]
        self.assertIn("ORDER BY c.embedding <=> %(vec)s", sql)
        self.assertIn("LIMIT %(limit)s", sql)
        self.assertNotIn("max_distance", sql)  # No min_similarity filter
        self.assertEqual(params["limit"], 3)
        self.assertIsInstance(params["vec"], Vector)

    def test_retrieve_with_min_similarity_adds_distance_filter(self) -> None:
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor

        with patch("app.services.retrieval.embed_query", return_value=[0.1] * 768):
            retrieval.retrieve(
                "slow wifi",
                top_k=5,
                min_similarity=0.7,
                conn=conn,
                ef_search=60,
            )

        # Verify ef_search local tune
        set_call = cursor.execute.call_args_list[0]
        self.assertEqual(set_call[0][0], "SET LOCAL hnsw.ef_search = %s;")
        self.assertEqual(set_call[0][1], (60,))

        # Verify query with max_distance = 1.0 - 0.7 = 0.3
        query_call = cursor.execute.call_args_list[1]
        sql, params = query_call[0]
        self.assertIn("AND (c.embedding <=> %(vec)s) <= %(max_distance)s", sql)
        self.assertAlmostEqual(params["max_distance"], 0.3)

    def test_retrieve_uses_managed_connection_when_conn_none(self) -> None:
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = cursor

        managed_ctx = MagicMock()
        managed_ctx.__enter__.return_value = mock_conn

        with (
            patch("app.services.retrieval.embed_query", return_value=[0.1] * 768),
            patch("app.services.retrieval.get_db_connection", return_value=managed_ctx),
        ):
            results = retrieval.retrieve("test question")

        self.assertEqual(results, [])
        managed_ctx.__enter__.assert_called_once()
        cursor.execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
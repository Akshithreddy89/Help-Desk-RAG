"""Tests for Gemini embedding service."""

from types import SimpleNamespace
import math
import unittest
from unittest.mock import MagicMock, patch

from app.services import embedding


def _fake_embed_response(texts: list[str], dims: int = 768) -> SimpleNamespace:
    items = []
    for i, _ in enumerate(texts):
        vec = [float(i + 1)] + [0.5] * (dims - 1)
        items.append(SimpleNamespace(values=vec))
    return SimpleNamespace(embeddings=items)


class EmbeddingTests(unittest.TestCase):
    """Verify document batching, query embedding, normalization, and validation."""

    def setUp(self) -> None:
        embedding.get_genai_client.cache_clear()

    def test_embed_documents_batches_and_normalizes(self) -> None:
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = (
            lambda model, contents, config: _fake_embed_response(contents, config.output_dimensionality)
        )

        with (
            patch.object(embedding.settings, "GEMINI_API_KEY", "test-key"),
            patch.object(embedding.settings, "EMBEDDING_MODEL", "gemini-embedding-001"),
            patch.object(embedding.settings, "EMBEDDING_DIMENSIONS", 768),
            patch("app.services.embedding.genai.Client", return_value=mock_client),
        ):
            vectors = embedding.embed_documents(
                ["chunk 1", "chunk 2", "chunk 3", "chunk 4", "chunk 5"],
                batch_size=2,
                title="wifi_troubleshooting.pdf",
            )

        self.assertEqual(len(vectors), 5)
        self.assertEqual(mock_client.models.embed_content.call_count, 3)

        first_call = mock_client.models.embed_content.call_args_list[0].kwargs
        self.assertEqual(first_call["model"], "gemini-embedding-001")
        self.assertEqual(first_call["contents"], ["chunk 1", "chunk 2"])
        self.assertEqual(first_call["config"].task_type, "RETRIEVAL_DOCUMENT")
        self.assertEqual(first_call["config"].output_dimensionality, 768)
        self.assertEqual(first_call["config"].title, "wifi_troubleshooting.pdf")

        for vec in vectors:
            self.assertEqual(len(vec), 768)
            norm = math.sqrt(sum(x * x for x in vec))
            self.assertAlmostEqual(norm, 1.0, places=6)

    def test_embed_query_uses_retrieval_query_task_type(self) -> None:
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = (
            lambda model, contents, config: _fake_embed_response(contents, config.output_dimensionality)
        )

        with (
            patch.object(embedding.settings, "GEMINI_API_KEY", "test-key"),
            patch.object(embedding.settings, "EMBEDDING_MODEL", "gemini-embedding-001"),
            patch.object(embedding.settings, "EMBEDDING_DIMENSIONS", 768),
            patch("app.services.embedding.genai.Client", return_value=mock_client),
        ):
            vec = embedding.embed_query("Why is my Wi-Fi slow?")

        self.assertEqual(len(vec), 768)
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        self.assertEqual(call_kwargs["contents"], ["Why is my Wi-Fi slow?"])
        self.assertEqual(call_kwargs["config"].task_type, "RETRIEVAL_QUERY")
        self.assertIsNone(call_kwargs["config"].title)

    def test_genai_client_has_bounded_timeout_and_no_retries(self) -> None:
        mock_client = MagicMock()
        with (
            patch.object(embedding.settings, "GEMINI_API_KEY", "test-key"),
            patch("app.services.embedding.genai.Client", return_value=mock_client) as client_factory,
        ):
            embedding.get_genai_client()

        options = client_factory.call_args.kwargs["http_options"]
        self.assertIsNone(options.timeout)
        self.assertEqual(options.retry_options.attempts, 1)

    def test_validation_errors(self) -> None:
        self.assertEqual(embedding.embed_documents([]), [])
        with self.assertRaises(ValueError):
            embedding.embed_documents(["   "])
        with self.assertRaises(ValueError):
            embedding.embed_documents(["hello"], batch_size=0)
        with self.assertRaises(ValueError):
            embedding.embed_query("   ")


if __name__ == "__main__":
    unittest.main()


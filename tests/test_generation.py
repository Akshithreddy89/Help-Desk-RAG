"""Unit tests for LLM answer generation service."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from app.services import generation
from app.services.retrieval import RetrievedChunk


def _make_dummy_chunk(
    index: int = 0,
    filename: str = "wifi_troubleshooting.pdf",
    page: int = 1,
    question: str = "Why is Wi-Fi slow?",
    content: str = "Move closer to router and check 5GHz band.",
    similarity: float = 0.85,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{index}",
        document_id="doc-1",
        filename=filename,
        chunk_index=index,
        content=content,
        similarity=similarity,
        distance=round(1.0 - similarity, 4),
        heading="Wi-Fi Troubleshooting",
        question=question,
        page_number=page,
        page_numbers=[page],
        metadata={"source": filename, "page_number": page},
    )


class GenerationTests(unittest.TestCase):
    """Verify grounded system prompt, greeting detection, scope rules, and fallback behavior."""

    def setUp(self) -> None:
        generation.get_llm_client.cache_clear()

    def test_empty_question_raises(self) -> None:
        with self.assertRaises(ValueError):
            generation.generate_answer("   ", [])

    def test_llm_client_has_bounded_timeout_and_no_retries(self) -> None:
        mock_client = MagicMock()
        with (
            patch.object(generation.settings, "LLM_API_KEY", "test-key"),
            patch("app.services.generation.genai.Client", return_value=mock_client) as client_factory,
        ):
            generation.get_llm_client()

        options = client_factory.call_args.kwargs["http_options"]
        self.assertIsNone(options.timeout)
        self.assertEqual(options.retry_options.attempts, 1)

    def test_detect_casual_greetings_fast_path(self) -> None:
        greetings = ["Hi", "Hello", "Good morning!", "thanks", "who are you?"]
        for g in greetings:
            reply = generation.detect_casual_greeting(g)
            self.assertIsNotNone(reply, f"Expected greeting reply for '{g}'")

        # Questions with technical intent should NOT be flagged as casual greetings
        tech_queries = ["Hello, why is my Wi-Fi slow?", "How do I fix Wi-Fi?", "router password"]
        for t in tech_queries:
            self.assertIsNone(generation.detect_casual_greeting(t), f"'{t}' should not be a greeting")

    def test_answer_question_returns_greeting_without_retrieval_or_llm(self) -> None:
        with (
            patch("app.services.generation.retrieve") as mock_retrieve,
            patch("app.services.generation.get_llm_client") as mock_client,
        ):
            res = generation.answer_question("Hello there!")

        self.assertTrue(res.is_greeting)
        self.assertIn("Helpdesk Assistant", res.answer)
        mock_retrieve.assert_not_called()
        mock_client.assert_not_called()

    def test_empty_chunks_returns_fallback_without_calling_llm(self) -> None:
        mock_client = MagicMock()
        result = generation.generate_answer(
            "How do I fix Wi-Fi?",
            [],
            client=mock_client,
        )

        self.assertEqual(result.question, "How do I fix Wi-Fi?")
        self.assertIn("could not find any relevant information", result.answer)
        self.assertEqual(result.sources, [])
        self.assertEqual(result.chunk_count, 0)
        self.assertFalse(result.is_greeting)
        mock_client.models.generate_content.assert_not_called()

    def test_format_context_blocks_includes_sources_and_content(self) -> None:
        chunks = [
            _make_dummy_chunk(0, page=1, question="Q1: Why is wifi slow?"),
            _make_dummy_chunk(1, page=2, question="Q2: How to forget network?"),
        ]
        context = generation.format_context_blocks(chunks)

        self.assertIn("[Source 1: wifi_troubleshooting.pdf | Page(s): 1]", context)
        self.assertIn("Section: Q1: Why is wifi slow?", context)
        self.assertIn("[Source 2: wifi_troubleshooting.pdf | Page(s): 2]", context)
        self.assertIn("Section: Q2: How to forget network?", context)
        self.assertIn("---", context)

    def test_generate_answer_calls_gemini_with_grounded_prompt(self) -> None:
        mock_response = SimpleNamespace(text="1. Move closer to the router [wifi_troubleshooting.pdf, Page 1].")
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        chunks = [_make_dummy_chunk(0)]

        with patch.object(generation.settings, "LLM_MODEL", "gemini-3.5-flash-lite"):
            result = generation.generate_answer(
                "Why is my Wi-Fi slow?",
                chunks,
                temperature=0.1,
                client=mock_client,
            )

        self.assertEqual(result.question, "Why is my Wi-Fi slow?")
        self.assertEqual(result.answer, "1. Move closer to the router [wifi_troubleshooting.pdf, Page 1].")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].filename, "wifi_troubleshooting.pdf")
        self.assertEqual(result.sources[0].page_numbers, [1])
        self.assertEqual(result.chunk_count, 1)

        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-3.5-flash-lite")
        self.assertIn("Why is my Wi-Fi slow?", call_kwargs["contents"])
        self.assertIn("wifi_troubleshooting.pdf", call_kwargs["contents"])

        config = call_kwargs["config"]
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.system_instruction, generation.GROUNDED_SYSTEM_INSTRUCTION)
        self.assertIn("Scope Restriction", config.system_instruction)

    def test_answer_question_chains_retrieval_and_generation(self) -> None:
        chunks = [_make_dummy_chunk(0)]
        fake_answer = generation.GeneratedAnswer(
            question="Why is Wi-Fi slow?",
            answer="Check router placement.",
            sources=[],
            model="gemini-3.5-flash-lite",
            chunk_count=1,
        )

        with (
            patch("app.services.generation.retrieve", return_value=chunks) as mock_retrieve,
            patch("app.services.generation.generate_answer", return_value=fake_answer) as mock_gen,
        ):
            out = generation.answer_question("Why is Wi-Fi slow?", top_k=3, min_similarity=0.6)

        mock_retrieve.assert_called_once_with(
            "Why is Wi-Fi slow?",
            top_k=3,
            min_similarity=0.6,
            conn=None,
        )
        mock_gen.assert_called_once_with(
            "Why is Wi-Fi slow?",
            chunks,
            model=None,
            temperature=0.2,
        )
        self.assertEqual(out.answer, "Check router placement.")


if __name__ == "__main__":
    unittest.main()

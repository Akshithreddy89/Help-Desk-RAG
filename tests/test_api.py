"""Unit tests for RAG chat API schemas and route handler."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from google.genai import errors as genai_errors
from pydantic import ValidationError

from app.api.chat import chat_endpoint
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource
from app.services.generation import GeneratedAnswer, GroundedSource


class ChatApiTests(unittest.TestCase):
    """Test POST /internal/rag/chat schemas, business logic, and route handler."""

    def setUp(self) -> None:
        self.mock_conn = MagicMock()

    @patch("app.api.chat.answer_question")
    def test_successful_rag_chat(self, mock_answer: MagicMock) -> None:
        mock_answer.return_value = GeneratedAnswer(
            question="How can I fix my weak WiFi signal?",
            answer="Try moving closer to your router and ensure line of sight.",
            sources=[
                GroundedSource(
                    filename="wifi_troubleshooting.pdf",
                    section="Weak WiFi Signal",
                    page_number=2,
                    page_numbers=[2],
                    similarity=0.88,
                )
            ],
            model="gemini-3.5-flash-lite",
            chunk_count=1,
            is_greeting=False,
        )

        request = ChatRequest(question="How can I fix my weak WiFi signal?")
        response = chat_endpoint(request, conn=self.mock_conn)

        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.answer, "Try moving closer to your router and ensure line of sight.")
        self.assertEqual(len(response.sources), 1)

        source_dict = response.sources[0].model_dump()
        self.assertEqual(
            source_dict,
            {
                "document": "wifi_troubleshooting.pdf",
                "section": "Weak WiFi Signal",
                "page_number": 2,
            },
        )
        mock_answer.assert_called_once_with(
            question="How can I fix my weak WiFi signal?",
            conn=self.mock_conn,
        )

    @patch("app.api.chat.answer_question")
    def test_greeting_chat_returns_no_sources(self, mock_answer: MagicMock) -> None:
        mock_answer.return_value = GeneratedAnswer(
            question="Hello",
            answer="Hello! I am your IT Helpdesk Assistant. How can I assist you today?",
            sources=[],
            model="gemini-3.5-flash-lite",
            chunk_count=0,
            is_greeting=True,
        )

        request = ChatRequest(question="Hello")
        response = chat_endpoint(request, conn=self.mock_conn)

        self.assertIsInstance(response, ChatResponse)
        self.assertIn("Helpdesk Assistant", response.answer)
        self.assertEqual(response.sources, [])

    def test_blank_question_raises_http_422(self) -> None:
        request = ChatRequest(question="     ")
        with self.assertRaises(HTTPException) as ctx:
            chat_endpoint(request, conn=self.mock_conn)

        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("cannot be blank", ctx.exception.detail)

    def test_schema_validations(self) -> None:
        # Empty string
        with self.assertRaises(ValidationError):
            ChatRequest(question="")

        # Missing question
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate({})

        # Disallow unexpected fields
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate({"question": "Valid?", "extra_field": "disallowed"})

    @patch("app.api.chat.answer_question")
    def test_internal_service_error_raises_http_500(self, mock_answer: MagicMock) -> None:
        mock_answer.side_effect = RuntimeError("Database connection timed out")

        request = ChatRequest(question="Why is Wi-Fi slow?")
        with self.assertRaises(HTTPException) as ctx:
            chat_endpoint(request, conn=self.mock_conn)

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("Failed to generate an answer", ctx.exception.detail)

    @patch("app.api.chat.answer_question")
    def test_gemini_deadline_error_raises_http_504(self, mock_answer: MagicMock) -> None:
        mock_answer.side_effect = genai_errors.ServerError(
            504,
            {
                "error": {
                    "code": 504,
                    "message": "Deadline expired before operation could complete.",
                    "status": "DEADLINE_EXCEEDED",
                }
            },
        )

        with self.assertRaises(HTTPException) as ctx:
            chat_endpoint(ChatRequest(question="Why is Wi-Fi slow?"), conn=self.mock_conn)

        self.assertEqual(ctx.exception.status_code, 504)
        self.assertNotIn("Gemini timed out", ctx.exception.detail)
        self.assertIn("did not complete", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
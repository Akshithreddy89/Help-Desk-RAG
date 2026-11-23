"""Internal RAG chat API endpoint."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from google.genai import errors as genai_errors

from app.database.connection import get_db
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource
from app.services.generation import answer_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/rag", tags=["RAG Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Internal RAG Chatbot Endpoint",
    description=(
        "Processes customer helpdesk questions using vector similarity retrieval "
        "and Gemini LLM generation, strictly grounded in the knowledge base."
    ),
)
def chat_endpoint(
    payload: ChatRequest,
    conn: Any = Depends(get_db),
) -> ChatResponse:
    """
    Handle incoming customer queries.

    1. Validates the question payload.
    2. Checks for fast-path casual greetings (0 tokens, 0 delay).
    3. Retrieves top relevant chunks from Supabase using pgvector cosine distance (<=>).
    4. Generates a strictly grounded response with source citations.
    5. Returns the answer and attributed sources.
    """
    cleaned_question = payload.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Question cannot be blank or contain only whitespace.",
        )

# call the internal rag generation pipeline

    try:
        result = answer_question(
            question=cleaned_question,
            conn=conn,
        )
    except genai_errors.APIError as exc:
        error_status = (
            status.HTTP_504_GATEWAY_TIMEOUT
            if exc.code == status.HTTP_504_GATEWAY_TIMEOUT
            else status.HTTP_502_BAD_GATEWAY
        )
        logger.warning("Gemini API request failed (%s): %s", exc.code, exc)
        raise HTTPException(
            status_code=error_status,
            detail="The upstream answer service did not complete the request.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in RAG chat endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate an answer. Please try again or contact support.",
        ) from exc


#FastAPI serializes this into clean JSON and sends an HTTP 200 OK back to Node.js
    sources = [
        ChatSource(
            document=s.filename,
            section=s.section,
            page_number=s.page_number,
        )
        for s in result.sources
    ]

    return ChatResponse(
        answer=result.answer,
        sources=sources,
    )

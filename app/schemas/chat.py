"""Pydantic schemas for the internal RAG chat endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChatSource(BaseModel):
    """Source citation indicating where an answer fact originated."""

    model_config = ConfigDict(extra="forbid")

    document: str = Field(
        ...,
        description="Source document filename",
        json_schema_extra={"example": "wifi_troubleshooting.pdf"},
    )
    section: str | None = Field(
        default=None,
        description="Section or FAQ question heading",
        json_schema_extra={"example": "Weak WiFi Signal"},
    )
    page_number: int | None = Field(
        default=None,
        description="Page number of the source document",
        json_schema_extra={"example": 2},
    )


class ChatRequest(BaseModel):
    """Customer query payload sent to the RAG service."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The customer's question or technical query",
        json_schema_extra={"example": "How can I fix my weak WiFi signal?"},
    )


class ChatResponse(BaseModel):
    """Grounded answer along with attributed sources."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        ...,
        description="Factual helpdesk answer generated from the knowledge base",
        json_schema_extra={"example": "Try moving closer to your router..."},
    )
    sources: list[ChatSource] = Field(
        default_factory=list,
        description="List of document sources cited in the answer",
    )


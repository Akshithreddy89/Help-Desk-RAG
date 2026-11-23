"""Text extraction service for knowledge base documents."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from app.config import BASE_DIR

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md"}) # 1) this are allowed 
DEFAULT_DOCUMENTS_DIR: Path = BASE_DIR / "knowledge_base" / "documents"  # from knowledge folder this will extract all the documents


@dataclass(frozen=True) #extracted data can't be accidentally modified
class ExtractedPage:   #extracted data for a single page (or entire plain-text document)
    """Extracted text for a single page (or entire plain-text document)."""

    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class ExtractedDocument: 
    """Extracted document text along with page-level text and metadata."""

    filename: str
    source_path: Path
    text: str
    pages: list[ExtractedPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

#5) clean text and remove extra spaces and line breaks and returtn clean text
def clean_text(text: str) -> str:
    """Normalize line breaks and collapse redundant whitespace while keeping paragraphs."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    joined = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()

#4) extract text from pdf and plain text files and return the extracted data
def _extract_pdf(path: Path) -> tuple[list[ExtractedPage], int]:
    """Extract cleaned text per page from a PDF using PyMuPDF."""
    pages: list[ExtractedPage] = []
    with pymupdf.open(path) as doc:
        page_count = len(doc)
        for page_number, page in enumerate(doc, start=1):
            cleaned = clean_text(page.get_text("text"))
            if cleaned:
                pages.append(ExtractedPage(text=cleaned, page_number=page_number))

    if not pages:
        logger.warning("No extractable text found in PDF '%s'.", path.name)

    return pages, page_count


#6) extract clean text from plain text
def _extract_plain_text(path: Path) -> list[ExtractedPage]:
    """Extract cleaned text from a UTF-8 text or Markdown file."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    if not cleaned:
        logger.warning("Document '%s' is empty after cleaning.", path.name)
        return []
    return [ExtractedPage(text=cleaned, page_number=None)]


#3) Extract document information
def extract_document(file_path: str | Path) -> ExtractedDocument:
    """
    Extract cleaned text and metadata from a supported document (.pdf, .txt, .md).

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If ``file_path`` is not a file or has an unsupported extension.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file path, got: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {supported}")

    if suffix == ".pdf":
        pages, page_count = _extract_pdf(path)
    else:
        pages = _extract_plain_text(path)
        page_count = 1 if pages else 0

    full_text = "\n\n".join(page.text for page in pages)
    metadata: dict[str, Any] = {
        "source": path.name,
        "file_type": suffix.lstrip("."),
        "page_count": page_count,
    }
    logger.info(
        "Extracted %d chars across %d non-empty page(s) from '%s'.",
        len(full_text),
        len(pages),
        path.name,
    )
    return ExtractedDocument(
        filename=path.name,
        source_path=path.resolve(),
        text=full_text,
        pages=pages,
        metadata=metadata,
    )

#6) Extract plain text from a supported document
def extract_text(file_path: str | Path) -> str:
    """Extract and return cleaned plain text from a supported document."""
    return extract_document(file_path).text


def list_knowledge_base_files(directory: str | Path | None = None) -> list[Path]:
    """Return sorted supported document paths inside the knowledge base folder."""
    docs_dir = Path(directory) if directory is not None else DEFAULT_DOCUMENTS_DIR
    if not docs_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {docs_dir}")

    return sorted(
        path
        for path in docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


#2) by this it find the knowledge base and find the documents and calls extract document function
def extract_knowledge_base(directory: str | Path | None = None) -> list[ExtractedDocument]:
    """Extract all supported documents (.pdf, .txt, .md) from the knowledge base."""
    return [extract_document(path) for path in list_knowledge_base_files(directory)]


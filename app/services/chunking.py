"""FAQ-aware chunking: one Q&A pair per chunk, split only if over the token limit."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.services.extraction import ExtractedDocument

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_QA_PREFIX_RE = re.compile(r"^(?:Q(?:uestion)?\s*\d*\s*[:.\-])\s*\S+", re.IGNORECASE)
_NUMBERED_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)*[.)]\s+\S+")
_BULLET_PREFIX_RE = re.compile(r"^[-*•●◦▪]\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass(frozen=True)
class TextChunk:
    """A single chunk of document text with structural and source metadata."""

    chunk_index: int
    content: str
    heading: str | None = None
    question: str | None = None
    section_type: str = "section"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _SectionBlock:
    """One heading section or one Q&A pair, with offsets into the source text."""

    text: str
    start: int
    end: int
    heading: str | None = None
    question: str | None = None
    section_type: str = "section"


def _token_count(text: str) -> int:
    """Approximate tokens as whitespace-separated words (no extra tokenizer dependency)."""
    return len(text.split()) if text else 0


#based on question 
def _is_question_line(line: str, next_line: str = "") -> bool:
    """Return True when a line starts a FAQ question."""
    if not line or _BULLET_PREFIX_RE.match(line):
        return False
    if _QA_PREFIX_RE.match(line):
        return True
    if _NUMBERED_PREFIX_RE.match(line):
        if line.endswith("?"):
            return True
        return bool(next_line) and not _BULLET_PREFIX_RE.match(next_line) and next_line.endswith("?")
    return line.endswith("?") and len(line) <= 200 and _token_count(line) <= 30


# based on heading
def _is_heading_line(line: str, next_line: str = "") -> bool:
    """Treat markdown headings as section labels; FAQ lines are not headings."""
    if not line or _BULLET_PREFIX_RE.match(line) or not _MD_HEADING_RE.match(line):
        return False
    return not _is_question_line(line, next_line)


def _clean_heading(line: str) -> str:
    match = _MD_HEADING_RE.match(line)
    return match.group(2).strip() if match else line.rstrip(":").strip()


#2) Divide into Headings & Q&A Blocks reads text line by line
def _split_into_sections(text: str) -> list[_SectionBlock]:
    """Split text into heading blocks and FAQ Q&A pairs in a single offset-tracked pass."""
    raw_lines = text.splitlines(keepends=True)
    blocks: list[_SectionBlock] = []
    current_heading: str | None = None
    current_question: str | None = None
    current_type = "section"
    buffer: list[str] = []
    buffer_start: int | None = None
    buffer_end = 0
    offset = 0
    i = 0
    n = len(raw_lines)

    def peek_stripped(index: int) -> str:
        return raw_lines[index].strip() if index < n else ""

    def flush_buffer() -> None:
        nonlocal buffer, buffer_start
        if buffer_start is None:
            buffer = []
            return
        body = "\n".join(buffer).strip()
        start, end = buffer_start, buffer_end
        buffer = []
        buffer_start = None
        if not body:
            return
        if current_type == "heading" and body.count("\n") == 0:
            return
        blocks.append(
            _SectionBlock(
                text=body,
                start=start,
                end=end,
                heading=current_heading,
                question=current_question,
                section_type=current_type,
            )
        )

    while i < n:
        raw_line = raw_lines[i]
        stripped = raw_line.strip()
        next_offset = offset + len(raw_line)
        next_line = peek_stripped(i + 1)

        if not stripped:
            if buffer and buffer[-1] != "":
                buffer.append("")
                buffer_end = next_offset
            offset = next_offset
            i += 1
            continue

        if _is_question_line(stripped, next_line):
            if current_heading is None and current_question is None and buffer:
                preamble = " ".join(ln for ln in buffer if ln)
                if _token_count(preamble) <= 25 and not any(_BULLET_PREFIX_RE.match(ln) for ln in buffer if ln):
                    current_heading = preamble
                    buffer = []
                    buffer_start = None
            flush_buffer()

            question_text = stripped
            consumed_end = next_offset
            if _NUMBERED_PREFIX_RE.match(stripped) and not stripped.endswith("?") and next_line.endswith("?"):
                question_text = f"{stripped} {next_line}"
                consumed_end += len(raw_lines[i + 1])
                i += 1

            current_question = question_text
            current_type = "faq"
            buffer = [question_text]
            buffer_start = offset
            buffer_end = consumed_end
            offset = consumed_end
            i += 1
            continue

        if _is_heading_line(stripped, next_line):
            flush_buffer()
            current_heading = _clean_heading(stripped)
            current_question = None
            current_type = "heading"
            buffer = [stripped]
            buffer_start = offset
            buffer_end = next_offset
            offset = next_offset
            i += 1
            continue

        buffer.append(stripped)
        if buffer_start is None:
            buffer_start = offset
        buffer_end = next_offset
        offset = next_offset
        i += 1

    flush_buffer()
    return blocks


def _split_long_unit(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split an oversized Q&A/section on sentence boundaries, with token overlap."""
    if _token_count(text) <= max_tokens:
        return [text.strip()]

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    units: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_tokens:
            units.append(sentence)
            continue
        step = max(max_tokens - overlap_tokens, 1)
        for start in range(0, len(words), step):
            units.append(" ".join(words[start : start + max_tokens]))
            if start + max_tokens >= len(words):
                break

    chunks: list[str] = []
    current_units: list[str] = []
    current_tokens = 0
    unit_sizes = [_token_count(unit) for unit in units]

    for unit, unit_tokens in zip(units, unit_sizes):
        if current_units and current_tokens + unit_tokens > max_tokens:
            chunks.append(" ".join(current_units).strip())
            overlap_buf: list[str] = []
            overlap_count = 0
            for prev in reversed(current_units):
                prev_len = _token_count(prev)
                if overlap_count + prev_len > overlap_tokens and overlap_buf:
                    break
                overlap_buf.insert(0, prev)
                overlap_count += prev_len
                if overlap_count >= overlap_tokens:
                    break
            current_units = overlap_buf
            current_tokens = overlap_count

        current_units.append(unit)
        current_tokens += unit_tokens

    if current_units:
        chunks.append(" ".join(current_units).strip())
    return chunks


def _validate_chunk_params(max_tokens: int, overlap_tokens: int) -> None:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")


def chunk_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    base_metadata: dict[str, Any] | None = None,
) -> list[TextChunk]:
    """
    Chunk document text as FAQ Q&A pairs (and markdown heading sections).

    Each question plus its answer is one chunk when it fits in ``max_tokens``.
    Oversized answers are split on sentences with ``overlap_tokens`` overlap,
    and the question/heading is repeated on continuation chunks.
    """
    _validate_chunk_params(max_tokens, overlap_tokens)
    if not text or not text.strip():
        return []

    extra_meta = base_metadata or {}
    results: list[TextChunk] = []

    for block in _split_into_sections(text):
        context_prefix = block.question if block.section_type == "faq" and block.question else block.heading or ""
        parts = _split_long_unit(block.text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

        for part_idx, part in enumerate(parts):
            content = part
            if part_idx > 0 and context_prefix and not content.startswith(context_prefix):
                content = f"{context_prefix}\n{content}"

            results.append(
                TextChunk(
                    chunk_index=len(results),
                    content=content,
                    heading=block.heading,
                    question=block.question,
                    section_type=block.section_type,
                    metadata={
                        **extra_meta,
                        "section_type": block.section_type,
                        "token_count": _token_count(content),
                        "char_start": block.start,
                        "char_end": block.end,
                        **({"heading": block.heading} if block.heading else {}),
                        **({"question": block.question} if block.question else {}),
                    },
                )
            )

    logger.info(
        "Chunked text into %d chunk(s) (max_tokens=%d, overlap_tokens=%d).",
        len(results),
        max_tokens,
        overlap_tokens,
    )
    return results


def _compute_page_spans(doc: ExtractedDocument) -> list[tuple[int, int, int]]:
    """Map character ranges [start, end) in ``doc.text`` back to PDF page numbers."""
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    last_index = len(doc.pages) - 1
    for i, page in enumerate(doc.pages):
        if page.page_number is None:
            cursor += len(page.text) + (2 if i < last_index else 0)
            continue
        start = cursor
        end = start + len(page.text)
        spans.append((start, end, page.page_number))
        cursor = end + (2 if i < last_index else 0)
    return spans


def _pages_for_span(spans: list[tuple[int, int, int]], start: int, end: int) -> list[int]:
    return [page for span_start, span_end, page in spans if span_start < end and span_end > start]


def chunk_document(
    doc: ExtractedDocument,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[TextChunk]:
    """
    Chunk an extracted document by Q&A pairs and attach contributing page numbers.

    Character offsets are recorded while splitting, so a question that starts on
    one page and finishes on the next stays one chunk with both page numbers.
    """
    chunks = chunk_text(
        doc.text,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        base_metadata=doc.metadata,
    )
    if not chunks or not doc.pages:
        return chunks

    page_spans = _compute_page_spans(doc)
    if not page_spans:
        return chunks

    enriched: list[TextChunk] = []
    for chunk in chunks:
        start = int(chunk.metadata.get("char_start", 0))
        end = int(chunk.metadata.get("char_end", start + len(chunk.content)))
        page_nums = _pages_for_span(page_spans, start, end)
        meta = dict(chunk.metadata)
        if page_nums:
            meta["page_number"] = page_nums[0]
            meta["page_numbers"] = page_nums
        enriched.append(
            TextChunk(
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                heading=chunk.heading,
                question=chunk.question,
                section_type=chunk.section_type,
                metadata=meta,
            )
        )
    return enriched

"""File ingestion helpers for job descriptions, candidate datasets, and schemas."""

from __future__ import annotations

import json
import logging
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import pandas as pd

LOGGER = logging.getLogger(__name__)


def read_job_description(name: str, content: bytes) -> str:
    """Read a PDF, DOCX, TXT, or MD job description into text."""

    suffix = Path(name).suffix.lower()
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix == ".docx":
            from docx import Document

            document = Document(BytesIO(content))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        return content.decode("utf-8", errors="replace")
    except Exception as exc:
        LOGGER.exception("Failed to read job description %s", name)
        raise ValueError(f"Could not parse job description {name}: {exc}") from exc


def read_schema(content: bytes) -> dict[str, Any]:
    """Read JSON Schema Draft 7 from uploaded bytes."""

    try:
        schema = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Candidate schema must be valid JSON: {exc}") from exc
    if not isinstance(schema, dict) or schema.get("type") not in {None, "object"}:
        raise ValueError("Candidate schema must describe an object.")
    return schema


def read_candidates(name: str, content: bytes | BinaryIO) -> list[dict[str, Any]]:
    """Read candidate records from JSON, JSONL, or CSV."""

    suffix = Path(name).suffix.lower()
    stream = BytesIO(content) if isinstance(content, bytes) else content
    normalized: list[dict[str, Any]] | None = None
    try:
        stream.seek(0)
        if suffix == ".jsonl":
            normalized = _read_jsonl(stream)
            rows = normalized
        elif suffix == ".json":
            parsed = json.loads(stream.read().decode("utf-8"))
            rows = parsed if isinstance(parsed, list) else parsed.get("candidates", [])
        elif suffix == ".csv":
            rows = pd.read_csv(stream).to_dict(orient="records")
        else:
            raise ValueError("Unsupported candidate format. Use JSON, JSONL, or CSV.")
    except Exception as exc:
        LOGGER.exception("Failed to parse candidates %s", name)
        raise ValueError(f"Could not parse candidate dataset {name}: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError("Candidate dataset must contain a list of records.")
    if normalized is None:
        normalized = [dict(row, _row_id=index + 1) for index, row in enumerate(rows) if isinstance(row, dict)]
    LOGGER.info("Loaded %s candidate records", len(normalized))
    return normalized


def iter_jsonl_candidates(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL candidate records from disk without loading the file into memory."""

    with Path(path).open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL on line {line_number}: {exc}") from exc
            if isinstance(row, dict):
                yield dict(row, _row_id=line_number)


def _read_jsonl(stream: BinaryIO) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(stream, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL on line {line_number}: {exc}") from exc
        if isinstance(row, dict):
            rows.append(dict(row, _row_id=len(rows) + 1))
    return rows

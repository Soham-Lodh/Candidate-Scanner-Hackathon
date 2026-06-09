"""Chunked upload service for large candidate datasets.

Streamlit's native file uploader is convenient, but it is not a streaming
endpoint: Tornado assembles the full request body before app code runs. This
small sidecar service accepts browser-sliced chunks and writes them directly to
disk so 400MB-500MB candidate JSONL files do not pass through Streamlit's
in-memory upload path.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)
UPLOAD_ROOT = Path("work") / "candidate_uploads"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
_SERVER: ThreadingHTTPServer | None = None
_LOCK = threading.Lock()
_UPLOADS: dict[str, "UploadState"] = {}
_SESSION_UPLOADS: dict[str, str] = {}


@dataclass(slots=True)
class UploadState:
    upload_id: str
    session_id: str
    filename: str
    path: str
    size: int
    received: int = 0
    complete: bool = False
    error: str = ""


def ensure_upload_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    """Start the local chunked upload service once and return its base URL."""

    global _SERVER
    if _SERVER is not None:
        return f"http://{host}:{port}"
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    _SERVER = ThreadingHTTPServer((host, port), _UploadHandler)
    thread = threading.Thread(target=_SERVER.serve_forever, daemon=True, name="candidate-upload-server")
    thread.start()
    LOGGER.info("Started candidate upload server on %s:%s", host, port)
    return f"http://{host}:{port}"


def latest_session_upload(session_id: str) -> UploadState | None:
    """Return the most recent upload state for a UI session."""

    with _LOCK:
        upload_id = _SESSION_UPLOADS.get(session_id)
        return _UPLOADS.get(upload_id) if upload_id else None


def reset_session_upload(session_id: str, *, delete_file: bool = False) -> None:
    """Clear the active upload pointer for a UI session."""

    with _LOCK:
        upload_id = _SESSION_UPLOADS.pop(session_id, None)
        state = _UPLOADS.pop(upload_id, None) if upload_id else None
    if delete_file and state:
        try:
            Path(state.path).unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("Could not delete upload %s", state.path)


def upload_status(upload_id: str) -> UploadState | None:
    """Return upload status by id."""

    with _LOCK:
        return _UPLOADS.get(upload_id)


class _UploadHandler(BaseHTTPRequestHandler):
    server_version = "CandidateUploadServer/1.0"

    def do_OPTIONS(self) -> None:
        self._send_empty(204)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        match = re.fullmatch(r"/uploads/session/([^/]+)", path)
        if match:
            state = latest_session_upload(match.group(1))
            self._send_json(_state_payload(state) if state else {"upload": None})
            return
        match = re.fullmatch(r"/uploads/([^/]+)", path)
        if match:
            state = upload_status(match.group(1))
            self._send_json(_state_payload(state) if state else {"error": "Upload not found"}, 200 if state else 404)
            return
        self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/uploads/start":
            self._start_upload()
            return
        match = re.fullmatch(r"/uploads/session/([^/]+)/reset", path)
        if match:
            reset_session_upload(match.group(1), delete_file=True)
            self._send_json({"ok": True})
            return
        match = re.fullmatch(r"/uploads/([^/]+)/chunk", path)
        if match:
            self._append_chunk(match.group(1))
            return
        match = re.fullmatch(r"/uploads/([^/]+)/complete", path)
        if match:
            self._complete_upload(match.group(1))
            return
        self._send_json({"error": "Not found"}, 404)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.debug("upload-server: " + format, *args)

    def _start_upload(self) -> None:
        try:
            payload = self._read_json()
            filename = Path(str(payload["filename"])).name
            session_id = str(payload["session_id"])
            size = int(payload["size"])
        except Exception:
            self._send_json({"error": "Invalid upload start payload"}, 400)
            return
        if size > MAX_UPLOAD_BYTES:
            self._send_json({"error": "Candidate upload must be 500MB or smaller"}, 413)
            return
        if Path(filename).suffix.lower() != ".jsonl":
            self._send_json({"error": "Large candidate uploads must be JSONL"}, 400)
            return
        upload_id = uuid.uuid4().hex
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
        path = UPLOAD_ROOT / f"{upload_id}_{safe_name}"
        path.write_bytes(b"")
        state = UploadState(
            upload_id=upload_id,
            session_id=session_id,
            filename=filename,
            path=str(path.resolve()),
            size=size,
        )
        with _LOCK:
            previous_id = _SESSION_UPLOADS.pop(session_id, None)
            previous_state = _UPLOADS.pop(previous_id, None) if previous_id else None
            _UPLOADS[upload_id] = state
            _SESSION_UPLOADS[session_id] = upload_id
        if previous_state:
            try:
                Path(previous_state.path).unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Could not delete superseded upload %s", previous_state.path)
        self._send_json(_state_payload(state))

    def _append_chunk(self, upload_id: str) -> None:
        with _LOCK:
            state = _UPLOADS.get(upload_id)
        if state is None:
            self._send_json({"error": "Upload not found"}, 404)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        expected_offset = int(self.headers.get("X-Chunk-Offset", "-1"))
        if content_length <= 0:
            self._send_json({"error": "Empty chunk"}, 400)
            return
        with _LOCK:
            if expected_offset != state.received:
                if 0 <= expected_offset < state.received:
                    self._send_json(_state_payload(state))
                else:
                    self._send_json({"error": f"Unexpected offset. Expected {state.received}."}, 409)
                return
            if state.received + content_length > MAX_UPLOAD_BYTES:
                state.error = "Upload exceeded 500MB limit"
                self._send_json({"error": state.error}, 413)
                return
        chunk = self.rfile.read(content_length)
        with Path(state.path).open("ab") as output:
            output.write(chunk)
        with _LOCK:
            state.received += len(chunk)
        self._send_json(_state_payload(state))

    def _complete_upload(self, upload_id: str) -> None:
        with _LOCK:
            state = _UPLOADS.get(upload_id)
            if state is None:
                self._send_json({"error": "Upload not found"}, 404)
                return
            if state.received != state.size:
                self._send_json({"error": f"Upload incomplete: {state.received}/{state.size} bytes"}, 409)
                return
            state.complete = True
        self._send_json(_state_payload(state))

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(content_length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Chunk-Offset")


def _state_payload(state: UploadState | None) -> dict[str, Any]:
    if state is None:
        return {"upload": None}
    payload = asdict(state)
    payload["percent"] = round((state.received / state.size) * 100, 2) if state.size else 0.0
    return {"upload": payload}

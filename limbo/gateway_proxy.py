"""Local retrying proxy in front of the model endpoint, for harness model traffic.

Harness CLIs give up on sustained HTTP 429 bursts after a few retries, which
ends the episode for reasons unrelated to the agent. The proxy retries the
upstream request (honoring Retry-After, capped) before the first byte is sent
back, then streams the upstream response through unchanged. Tool calls do not
pass through here; only model API requests do. The upstream is ``LIMBO_BASE_URL``.
"""

from __future__ import annotations

import random
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .llm import resolve_credentials

RETRY_STATUS = {429, 500, 502, 503, 504}
HOP_HEADERS = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}

_lock = threading.Lock()
_server: ThreadingHTTPServer | None = None


class _Handler(BaseHTTPRequestHandler):
    upstream = ""
    max_attempts = 14

    def log_message(self, *a):
        pass

    def _forward(self, method: str) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        url = self.upstream.rstrip("/") + self.path
        resp = None
        for attempt in range(self.max_attempts):
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                resp = urllib.request.urlopen(req, timeout=300)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_STATUS or attempt == self.max_attempts - 1:
                    resp = exc
                    break
                ra = exc.headers.get("retry-after") if exc.headers else None
                wait = float(ra) if ra and ra.strip().isdigit() else min(30.0, 1.5 * 2 ** attempt)
                exc.read()
                time.sleep(min(wait, 60.0) * (0.8 + 0.4 * random.random()))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                if attempt == self.max_attempts - 1:
                    self.send_error(502, "upstream unreachable")
                    return
                time.sleep(min(30.0, 1.5 * 2 ** attempt))
        status = getattr(resp, "status", None) or getattr(resp, "code", 502)
        self.send_response(status)
        for k, v in resp.headers.items():
            if k.lower() not in HOP_HEADERS:
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        while True:
            chunk = resp.read1(8192) if hasattr(resp, "read1") else resp.read(8192)
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break
        self.close_connection = True

    def do_POST(self):
        self._forward("POST")

    def do_GET(self):
        self._forward("GET")


def ensure_proxy() -> str:
    """Start (once per process) and return the proxy base URL."""
    global _server
    with _lock:
        if _server is None:
            _Handler.upstream = resolve_credentials()[0]
            _server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            _server.daemon_threads = True
            threading.Thread(target=_server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{_server.server_address[1]}"

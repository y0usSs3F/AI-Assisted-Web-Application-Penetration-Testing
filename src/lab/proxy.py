"""
proxy.py -- a logging reverse proxy for the practical lab.

    client (benign user / OWASP ZAP) --> 127.0.0.1:3001 (this proxy) --> 127.0.0.1:3000 (Juice Shop)

Every request is forwarded unchanged and written to a JSON-lines log with the current lab
phase. Active-scan requests from ZAP carry the header X-ZAP-Scan-ID (ZAP's scan-rule id),
which gives per-request ground truth; the analyzer never sees that header.
The proxy listens on the loopback interface only.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "proxy-connection"}
MAX_BODY_LOG = 20_000


class LabProxy:
    def __init__(self, upstream_port=3000, listen_port=3001, log_path=None):
        self.upstream_port, self.listen_port, self.log_path = upstream_port, listen_port, log_path
        self.phase = "idle"
        self.count = 0
        self._lock = threading.Lock()
        self._log = open(log_path, "a", encoding="utf-8") if log_path else None
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def __getattr__(self, name):          # handle any HTTP method ZAP may send
                if name.startswith("do_"):
                    return self._proxy
                raise AttributeError(name)

            def _proxy(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n > 0 else b""
                path, _, query = self.path.partition("?")
                if path.startswith("http://") or path.startswith("https://"):
                    path = "/" + path.split("/", 3)[3] if path.count("/") >= 3 else "/"
                hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
                status, rbody, rhdrs = 502, b"", []
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", proxy.upstream_port, timeout=60)
                    hdrs["Host"] = f"127.0.0.1:{proxy.upstream_port}"
                    conn.request(self.command, self.path if not self.path.startswith("http") else
                                 (path + ("?" + query if query else "")), body=body or None, headers=hdrs)
                    resp = conn.getresponse()
                    status, rbody, rhdrs = resp.status, resp.read(), resp.getheaders()
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
                proxy._record(self.command, path, query, dict(self.headers.items()), body, status, len(rbody))
                try:
                    self.send_response(status)
                    for k, v in rhdrs:
                        if k.lower() not in HOP and k.lower() != "content-length":
                            self.send_header(k, v)
                    self.send_header("Content-Length", str(len(rbody)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(rbody)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", listen_port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def _record(self, method, path, query, headers, body, status, size):
        rec = {"ts": time.time(), "phase": self.phase, "method": method, "path": path,
               "query": query, "headers": headers,
               "body": body[:MAX_BODY_LOG].decode("utf-8", errors="replace"),
               "status": status, "response_bytes": size,
               "zap_scan_id": next((v for k, v in headers.items() if k.lower() == "x-zap-scan-id"), None)}
        with self._lock:
            self.count += 1
            if self._log:
                self._log.write(json.dumps(rec) + "\n")
                self._log.flush()

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        if self._log:
            self._log.close()

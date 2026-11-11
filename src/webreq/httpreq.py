"""
httpreq.py
A small, dependency-free representation of an HTTP request and parsers for the formats the
project meets: the raw text of HTTP CSIC 2010, raw requests pasted into the UI, and the JSON
lines written by the lab's logging proxy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, unquote_plus, urlsplit

METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS")
_START = re.compile(r"^(%s) \S+" % "|".join(METHODS))

# Headers that identify a tool or a session rather than describe the request content.
# They are kept for display but never used as model input, so a detector cannot learn
# "this is the scanner's User-Agent" instead of "this is an attack" (cf. the TCP-window
# shortcut found in the CIC-IDS2017 audit).
IGNORED_HEADERS = {"user-agent", "cookie", "host", "content-length", "connection",
                   "accept", "accept-encoding", "accept-charset", "accept-language",
                   "authorization", "pragma", "cache-control", "referer", "origin",
                   "x-zap-scan-id", "sec-fetch-site", "sec-fetch-mode", "sec-fetch-dest",
                   "if-none-match", "if-modified-since", "content-type", "upgrade-insecure-requests"}


@dataclass
class Request:
    method: str
    path: str                     # URL path without scheme/host, still URL-encoded
    query: str = ""               # raw query string (URL-encoded)
    body: str = ""                # raw body
    headers: dict = field(default_factory=dict)   # lower-case name -> value

    # -------------------------------------------------------------- derived views
    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").lower()

    def params(self) -> list[tuple[str, str, str]]:
        """(location, name, decoded value) for every query, form, JSON and path parameter."""
        out = [("query", k, v) for k, v in parse_qsl(self.query, keep_blank_values=True)]
        body = self.body.strip()
        if body:
            if body[:1] in "{[" or "json" in self.content_type:
                out += [("json", k, v) for k, v in _json_leaves(body)]
            elif "=" in body and "multipart" not in self.content_type:
                out += [("body", k, v) for k, v in parse_qsl(body, keep_blank_values=True)]
            else:
                out.append(("body", "<raw>", unquote_plus(body)))
        # REST-style path segments that carry unusual data (e.g. /rest/products/1'--/reviews)
        segs = [unquote_plus(s) for s in self.path.split("/") if s]
        for i, s in enumerate(segs, start=1):
            if not re.fullmatch(r"[A-Za-z0-9_.~-]+", s):
                out.append(("path", f"seg{i}", s))
        full = unquote_plus(unquote_plus(self.path))
        if ".." in full or "\\" in full or "\x00" in full:  # traversal hides in "safe" segments
            out.append(("path", "<path>", full))
        return out

    def header_values(self) -> list[str]:
        """Values of content-bearing headers (tool and session headers excluded)."""
        return [v for k, v in self.headers.items() if k not in IGNORED_HEADERS]

    def endpoint(self) -> str:
        """Path with numeric/identifier segments collapsed, for grouping findings."""
        segs = []
        for s in self.path.split("/"):
            d = unquote_plus(s)
            if re.fullmatch(r"\d+|[0-9a-f]{8,}|[0-9a-f-]{12,}", d):
                s = "{id}"
            elif d and not re.fullmatch(r"[A-Za-z0-9_.~-]+", d) or d == "..":
                s = "{value}"                       # data (or an attack payload) in the path
            segs.append(s)
        return f"{self.method} {'/'.join(segs) or '/'}"

    def to_text(self) -> str:
        """Decoded request line + body: the input of the character n-gram (NLP) models."""
        q = f"?{unquote_plus(self.query)}" if self.query else ""
        return f"{self.method} {unquote_plus(self.path)}{q} {unquote_plus(self.body)}".lower()

    def to_raw(self) -> str:
        q = f"?{self.query}" if self.query else ""
        hdrs = "".join(f"{k.title()}: {v}\n" for k, v in self.headers.items())
        return f"{self.method} {self.path}{q} HTTP/1.1\n{hdrs}\n{self.body}"

    def to_dict(self) -> dict:
        return {"method": self.method, "path": self.path, "query": self.query,
                "body": self.body, "headers": dict(self.headers)}

    @classmethod
    def from_dict(cls, d: dict) -> "Request":
        return cls(method=d.get("method", "GET").upper(), path=d.get("path", "/") or "/",
                   query=d.get("query", "") or "", body=d.get("body", "") or "",
                   headers={k.lower(): str(v) for k, v in (d.get("headers") or {}).items()})


def _json_leaves(body: str):
    import json
    try:
        obj = json.loads(body)
    except ValueError:
        return [("<raw>", body)]
    out = []

    def walk(o, key):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{key}.{k}" if key else str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{key}[{i}]")
        elif o is not None:
            out.append((key or "<root>", str(o)))
    walk(obj, "")
    return out


def from_url(method: str, url: str, headers: dict | None = None, body: str = "") -> Request:
    u = urlsplit(url)
    return Request(method=method.upper(), path=u.path or "/", query=u.query, body=body,
                   headers={k.lower(): v for k, v in (headers or {}).items()})


def parse_raw(raw: str) -> Request:
    """Parse one raw HTTP request: request line, headers, blank line, body."""
    raw = raw.replace("\r\n", "\n").lstrip("\n")
    head, _, body = raw.partition("\n\n")
    lines = head.split("\n")
    parts = lines[0].split(" ")
    method = parts[0].upper() if parts else "GET"
    target = parts[1] if len(parts) > 1 else "/"
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    clen = headers.get("content-length")
    body = body.strip("\n")
    if clen and clen.isdigit():
        body = body[: int(clen)]
    return from_url(method, target, headers, body)


def split_raw_stream(text: str) -> list[str]:
    """Split a text file of concatenated raw requests (the CSIC 2010 format)."""
    text = text.replace("\r\n", "\n")
    chunks, cur = [], []
    for line in text.split("\n"):
        if _START.match(line) and line.rstrip().endswith("HTTP/1.1") and cur:
            chunks.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur and any(c.strip() for c in cur):
        chunks.append("\n".join(cur))
    return [c for c in chunks if _START.match(c)]

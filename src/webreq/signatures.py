"""
signatures.py
The conventional, signature-based baseline for request-level detection.

A web application firewall (WAF) or IDS signature set matches decoded request content against
hand-written patterns. The rules below are written for this project and modelled on the rule
families of the OWASP ModSecurity Core Rule Set (CRS) at paranoia level 1:
942 SQL injection, 941 XSS, 930 path traversal / local file inclusion, 932 remote command
execution, plus a few generic injection rules. They need no training data, every alert names
the rule that fired, and each rule implies an attack class, so the baseline can both detect
and classify. It cannot recognise attacks for which no rule exists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from webreq.features import full_decode
from webreq.httpreq import Request


@dataclass(frozen=True)
class Rule:
    rid: str
    category: str      # sqli | xss | path-traversal | cmdi | other
    description: str
    pattern: re.Pattern


def _r(rid, cat, desc, rx):
    return Rule(rid, cat, desc, re.compile(rx, re.I | re.S))


RULES = [
    # --- SQL injection (CRS 942xxx family) ---------------------------------------------
    _r("SQLI-01", "sqli", "UNION-based query", r"\bunion\b[\s\S]{0,40}\bselect\b"),
    _r("SQLI-02", "sqli", "tautology after quote", r"['\"\)]\s*(or|and|\|\||&&)\s+['\"(]?\w+['\"]?\s*(=|like|<|>)"),
    _r("SQLI-03", "sqli", "stacked query", r";\s*(select|insert|update|delete|drop|create|alter|exec|declare|shutdown)\b"),
    _r("SQLI-04", "sqli", "time-based blind", r"\b(sleep|pg_sleep|benchmark)\s*\(|\bwaitfor\s+delay\b"),
    _r("SQLI-05", "sqli", "comment after quote", r"['\"]\s*(--|#|/\*)"),
    _r("SQLI-06", "sqli", "error-based function", r"\b(extractvalue|updatexml|exp|floor\s*\(\s*rand)\s*\("),
    _r("SQLI-07", "sqli", "schema enumeration", r"\b(information_schema|sysobjects|sqlite_master|pg_catalog|mysql\.user)\b"),
    _r("SQLI-08", "sqli", "SELECT ... FROM", r"\bselect\b[\s\S]{1,80}\bfrom\b"),
    _r("SQLI-09", "sqli", "ORDER BY / GROUP BY probe", r"['\")]\s*(order|group)\s+by\s+\d+"),
    _r("SQLI-10", "sqli", "dangerous procedure", r"\b(xp_cmdshell|sp_executesql|load_file|into\s+(out|dump)file)\b"),
    _r("SQLI-11", "sqli", "boolean probe", r"\b(and|or)\s+\d+\s*=\s*\d+"),
    # --- Cross-site scripting (CRS 941xxx family) ---------------------------------------
    _r("XSS-01", "xss", "script tag", r"<\s*/?\s*script\b"),
    _r("XSS-02", "xss", "event handler attribute", r"<[^>]*\bon[a-z]{3,}\s*="),
    _r("XSS-03", "xss", "javascript: URI", r"(java|vb)script\s*:"),
    _r("XSS-04", "xss", "dangerous HTML element", r"<\s*(iframe|object|embed|svg|math|applet|base|meta|link|frameset)\b"),
    _r("XSS-05", "xss", "DOM sink", r"document\s*\.\s*(cookie|domain|write|location)|window\s*\.\s*location"),
    _r("XSS-06", "xss", "JS execution function", r"\b(alert|prompt|confirm|eval)\s*\(|fromcharcode"),
    _r("XSS-07", "xss", "img/src injection", r"<\s*img\b[^>]*\bsrc\s*="),
    # --- Path traversal / LFI (CRS 930xxx family) --------------------------------------
    _r("LFI-01", "path-traversal", "directory traversal", r"(\.\.[/\\]){1,}|(%2e%2e|\.%2e|%2e\.)[/\\%]"),
    _r("LFI-02", "path-traversal", "sensitive OS file", r"(/etc/(passwd|shadow|group|hosts|issue)|boot\.ini|win\.ini|system32)"),
    _r("LFI-03", "path-traversal", "file / php wrapper", r"\b(file|php|zip|data|expect|phar)://"),
    _r("LFI-04", "path-traversal", "/proc access", r"/proc/(self|\d+)/"),
    # --- OS command injection (CRS 932xxx family) --------------------------------------
    _r("RCE-01", "cmdi", "command after separator", r"[;|&`]\s*(ls|cat|id|whoami|uname|pwd|ifconfig|ipconfig|netstat|ping|nslookup|wget|curl|nc|bash|sh|powershell|cmd)\b"),
    _r("RCE-02", "cmdi", "command substitution", r"\$\([^)]*\)|`[^`]+`"),
    _r("RCE-03", "cmdi", "shell path", r"/bin/(ba|z|da)?sh\b|cmd(\.exe)?\s+/c\b"),
    # --- Other injection --------------------------------------------------------------
    _r("GEN-01", "other", "null byte", r"%00|\x00"),
    _r("GEN-02", "other", "server-side include", r"<!--\s*#\s*(exec|include|echo)"),
    _r("GEN-03", "other", "template / expression injection", r"\{\{[\s\S]*\}\}|\$\{[\s\S]*\}"),
    _r("GEN-04", "other", "XML external entity", r"<!entity|<!doctype[^>]*\["),
    _r("GEN-05", "other", "CRLF injection", r"%0d%0a|\r\n[a-z-]+\s*:"),
    _r("GEN-06", "other", "LDAP injection", r"\*\)\(|\)\(\|"),
]
CATEGORIES = ["sqli", "xss", "path-traversal", "cmdi", "other"]


def inspect_strings(strings) -> list[Rule]:
    hits = []
    for s in strings:
        d = full_decode(s)
        for rule in RULES:
            if rule not in hits and rule.pattern.search(d):
                hits.append(rule)
    return hits


def inspect(req: Request) -> list[Rule]:
    """Rules that fire on any parameter value, the path, or a content-bearing header."""
    strings = [v for _, _, v in req.params()] + [req.path] + req.header_values()
    return inspect_strings(strings)


class SignatureDetector:
    """WAF-style detector: alert if any rule fires; the attack class is the majority category."""

    name = "Signature rules (conventional)"

    def fit(self, *_):
        return self

    def detect(self, reqs):
        hits = [inspect(r) for r in reqs]
        pred = [int(bool(h)) for h in hits]
        cats = [_majority(h) for h in hits]
        return pred, cats, hits

    def classify_values(self, values) -> list[str]:
        """Attack class of bare parameter values (HttpParamsDataset); 'norm' if no rule fires."""
        return [_majority(inspect_strings([v])) or "norm" for v in values]


def _majority(hits) -> str | None:
    if not hits:
        return None
    counts = {}
    for r in hits:
        counts[r.category] = counts.get(r.category, 0) + 1
    return max(counts, key=lambda c: (counts[c], -CATEGORIES.index(c)))

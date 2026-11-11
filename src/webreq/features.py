"""
features.py
Interpretable request features, grouped the way the project brief asks for them:
URL structure, parameters, headers and request content.

Every value is decoded before inspection (URL-decoding applied up to twice, so that
double-encoded payloads such as %2527 are seen as a quote). Tool- and session-identifying
headers (User-Agent, Cookie, Host, ...) are excluded on purpose: a detector must not learn
to recognise a scanner instead of an attack.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import unquote

import numpy as np
import pandas as pd

from webreq.httpreq import Request

# ------------------------------------------------------------------ keyword lexicons
SQL_KW = re.compile(r"\b(union|select|insert|update|delete|drop|alter|from|where|having|"
                    r"order\s+by|group\s+by|sleep|benchmark|waitfor|delay|information_schema|"
                    r"concat|char|ascii|substring|extractvalue|updatexml|pg_sleep|xp_cmdshell|"
                    r"exec|declare|cast|convert|null|or|and)\b", re.I)
SQL_TAUT = re.compile(r"['\"\)]\s*(or|and|\|\||&&)\s*['\"(]?\s*[\w'\"]+\s*['\"]?\s*(=|like|<|>)", re.I)
SQL_COMMENT = re.compile(r"(--|#|/\*|\*/|;\s*$)")
XSS_KW = re.compile(r"(<\s*script|javascript:|vbscript:|on\w+\s*=|alert\s*\(|prompt\s*\(|"
                    r"confirm\s*\(|document\.(cookie|location|write)|<\s*iframe|<\s*img|<\s*svg|"
                    r"<\s*body|<\s*object|<\s*embed|eval\s*\(|fromcharcode|expression\s*\(|"
                    r"srcdoc|<\s*a\s+href)", re.I)
TRAV_KW = re.compile(r"(\.\./|\.\.\\|/etc/(passwd|shadow|hosts)|boot\.ini|win\.ini|"
                     r"c:\\|/proc/self|file://|%2e%2e)", re.I)
CMD_KW = re.compile(r"([;|&`]\s*(ls|cat|id|whoami|uname|pwd|dir|type|ping|nslookup|wget|curl|nc|"
                    r"bash|sh|cmd|powershell|echo|sleep)\b|\$\(|/bin/(ba)?sh|cmd\.exe|&&)", re.I)
OTHER_KW = re.compile(r"(<!--#|\{\{|\$\{|%00|\x00|\r\n|<!entity|<!doctype\s+\w+\s*\[|"
                      r"\(\|\(|\)\(|\*\)\()", re.I)
SUSPICIOUS_EXT = re.compile(r"\.(bak|old|orig|backup|inc|conf|config|ini|log|sql|tmp|swp|zip|tar|gz|"
                            r"git|env|htaccess|htpasswd|DS_Store|~)$|~$", re.I)
SPECIAL = set("'\"<>;()=*%$|&`{}[]\\!")
# engineered: obfuscation / evasion encodings and code-like structure
OBFUSCATION = re.compile(r"(\\x[0-9a-f]{2}|\\u[0-9a-f]{4}|%u[0-9a-f]{4}|0x[0-9a-f]{4,}|&#x?[0-9a-f]+;?|"
                         r"\b(char|chr|unhex|hex|base64_decode|atob|string\.fromcharcode)\s*\(|/\*.*?\*/)", re.I)
FUNC_CALL = re.compile(r"[a-z_][a-z0-9_.]*\s*\(", re.I)
URL_IN_VALUE = re.compile(r"(https?|ftp|file|php|data|gopher)://|\bwww\.", re.I)
KEYWORDS = re.compile("|".join(p.pattern for p in (SQL_KW, XSS_KW, CMD_KW)), re.I)

# Features in display order, with the group each belongs to (used by the UI and paper).
GROUPS = {
    "url": ["method_post", "method_other", "path_len", "path_depth", "path_nonalnum",
            "path_suspicious_ext", "path_traversal"],
    "parameters": ["n_params", "query_len", "body_len", "max_value_len", "mean_value_len",
                   "pct_encoded", "double_encoded"],
    "headers": ["header_special", "header_keywords"],
    "content": ["n_special", "n_quotes", "n_angle", "n_parens", "n_semicolon", "n_comment",
                "n_pipe_amp", "n_spaces", "digit_ratio", "alpha_ratio", "upper_ratio",
                "nonascii", "entropy", "kw_sql", "sql_tautology", "kw_xss", "kw_traversal",
                "kw_cmd", "kw_other"],
}
BASE_FEATURES = [f for g in GROUPS.values() for f in g]
# Engineered content features (added after the exploratory analysis, see webreq/tuning.py):
# they describe *how* a value is built rather than which keywords it contains, so they
# also catch obfuscated, mixed-case or keyword-free payloads.
ENGINEERED = ["max_special_run", "n_func_calls", "class_switch_rate", "n_obfuscation",
              "mixed_case_kw", "kw_density", "n_tokens", "nonalnum_ratio", "url_in_value"]
GROUPS["content"] = GROUPS["content"] + ENGINEERED
FEATURES = BASE_FEATURES + ENGINEERED
FEATURE_GROUP = {f: g for g, fs in GROUPS.items() for f in fs}
LABELS = {
    "method_post": "POST method", "method_other": "PUT/DELETE/other method",
    "path_len": "URL path length", "path_depth": "URL path depth",
    "path_nonalnum": "unusual characters in path", "path_suspicious_ext": "backup/config file extension",
    "path_traversal": "'../' in path", "n_params": "number of parameters",
    "query_len": "query string length", "body_len": "body length",
    "max_value_len": "longest parameter value", "mean_value_len": "mean parameter length",
    "pct_encoded": "%-encoded characters", "double_encoded": "double URL-encoding",
    "header_special": "special characters in headers", "header_keywords": "attack keywords in headers",
    "n_special": "special characters", "n_quotes": "quotes", "n_angle": "angle brackets < >",
    "n_parens": "parentheses", "n_semicolon": "semicolons", "n_comment": "SQL comment markers",
    "n_pipe_amp": "pipes and ampersands", "n_spaces": "spaces", "digit_ratio": "share of digits",
    "alpha_ratio": "share of letters", "upper_ratio": "share of upper-case",
    "nonascii": "non-ASCII characters", "entropy": "character entropy",
    "kw_sql": "SQL keywords", "sql_tautology": "SQL tautology (' or 1=1)",
    "kw_xss": "script/HTML injection keywords", "kw_traversal": "path-traversal patterns",
    "kw_cmd": "OS command patterns", "kw_other": "SSI/template/null-byte/XXE patterns",
    "max_special_run": "longest run of special characters", "n_func_calls": "function-call patterns f(",
    "class_switch_rate": "character-class switches per character", "n_obfuscation": "encoding/obfuscation tricks",
    "mixed_case_kw": "mixed-case keywords (SeLeCt)", "kw_density": "attack keywords per token",
    "n_tokens": "number of tokens", "nonalnum_ratio": "share of non-alphanumeric characters",
    "url_in_value": "URL inside a value",
}


def full_decode(s: str, rounds: int = 2) -> str:
    for _ in range(rounds):
        d = unquote(s)
        if d == s:
            break
        s = d
    return s


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _cls(ch: str) -> int:
    return 0 if ch.isalpha() else 1 if ch.isdigit() else 2 if ch.isspace() else 3


def _max_run(text: str) -> int:
    best = cur = 0
    for ch in text:
        cur = cur + 1 if ch in SPECIAL else 0
        best = max(best, cur)
    return best


def engineered_features(text: str) -> dict:
    n = max(len(text), 1)
    tokens = [t for t in re.split(r"[^A-Za-z0-9_]+", text) if t]
    kws = [m.group(0) for m in KEYWORDS.finditer(text)]
    mixed = sum(1 for k in kws if any(c.isupper() for c in k) and any(c.islower() for c in k)
                and not (k[:1].isupper() and k[1:].islower()))
    return {
        "max_special_run": _max_run(text),
        "n_func_calls": len(FUNC_CALL.findall(text)),
        "class_switch_rate": sum(_cls(a) != _cls(b) for a, b in zip(text, text[1:])) / n,
        "n_obfuscation": len(OBFUSCATION.findall(text)),
        "mixed_case_kw": mixed,
        "kw_density": len(kws) / max(len(tokens), 1),
        "n_tokens": len(tokens),
        "nonalnum_ratio": sum(not (c.isalnum() or c.isspace()) for c in text) / n,
        "url_in_value": len(URL_IN_VALUE.findall(text)),
    }


def content_features(text: str) -> dict:
    """Content features of one decoded string (a parameter value or a whole payload)."""
    n = max(len(text), 1)
    return {
        "n_special": sum(ch in SPECIAL for ch in text),
        "n_quotes": text.count("'") + text.count('"'),
        "n_angle": text.count("<") + text.count(">"),
        "n_parens": text.count("(") + text.count(")"),
        "n_semicolon": text.count(";"),
        "n_comment": len(SQL_COMMENT.findall(text)),
        "n_pipe_amp": text.count("|") + text.count("&"),
        "n_spaces": text.count(" "),
        "digit_ratio": sum(ch.isdigit() for ch in text) / n,
        "alpha_ratio": sum(ch.isalpha() for ch in text) / n,
        "upper_ratio": sum(ch.isupper() for ch in text) / n,
        "nonascii": sum(ord(ch) > 127 for ch in text),
        "entropy": _entropy(text),
        "kw_sql": len(SQL_KW.findall(text)),
        "sql_tautology": len(SQL_TAUT.findall(text)),
        "kw_xss": len(XSS_KW.findall(text)),
        "kw_traversal": len(TRAV_KW.findall(text)),
        "kw_cmd": len(CMD_KW.findall(text)),
        "kw_other": len(OTHER_KW.findall(text)),
    } | engineered_features(text)


def extract(req: Request) -> dict:
    params = req.params()
    values = [full_decode(v) for _, _, v in params]
    content = " ".join(values)
    path = full_decode(req.path)
    raw = req.path + req.query + req.body
    hvals = " ".join(full_decode(v) for v in req.header_values())
    f = {
        "method_post": int(req.method == "POST"),
        "method_other": int(req.method not in ("GET", "POST")),
        "path_len": len(path),
        "path_depth": path.count("/"),
        "path_nonalnum": len(re.findall(r"[^A-Za-z0-9/._~-]", path)),
        "path_suspicious_ext": int(bool(SUSPICIOUS_EXT.search(path.rstrip("/")))),
        "path_traversal": len(TRAV_KW.findall(path)),
        "n_params": len(params),
        "query_len": len(req.query),
        "body_len": len(req.body),
        "max_value_len": max((len(v) for v in values), default=0),
        "mean_value_len": float(np.mean([len(v) for v in values])) if values else 0.0,
        "pct_encoded": len(re.findall(r"%[0-9a-fA-F]{2}", raw)),
        "double_encoded": len(re.findall(r"%25[0-9a-fA-F]{2}", raw)),
        "header_special": sum(ch in SPECIAL for ch in hvals),
        "header_keywords": (len(SQL_TAUT.findall(hvals)) + len(XSS_KW.findall(hvals))
                            + len(TRAV_KW.findall(hvals)) + len(CMD_KW.findall(hvals))),
    }
    f.update(content_features(content))
    return f


def frame(reqs) -> pd.DataFrame:
    return pd.DataFrame([extract(r) for r in reqs], columns=FEATURES)


def payload_request(value: str, name: str = "q") -> Request:
    """Wrap a bare parameter value (HttpParamsDataset) as a one-parameter GET request."""
    from urllib.parse import quote
    return Request(method="GET", path="/search", query=f"{name}={quote(value, safe='')}")

"""
parse_zap_report.py
Normalise an OWASP ZAP JSON report into a flat findings table, with each alert mapped to
an attack category via its CWE id (or its name as a fallback).

  python parse_zap_report.py lab_run/zap_report.json lab_run/zap_findings.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

CWE_MAP = {89: "SQL Injection", 79: "XSS", 22: "Path Traversal", 78: "Command Injection",
           77: "Command Injection", 307: "Brute Force / Auth", 287: "Brute Force / Auth"}
NAME_KEYWORDS = [("sql", "SQL Injection"), ("cross site scripting", "XSS"), ("xss", "XSS"),
                 ("path traversal", "Path Traversal"), ("command injection", "Command Injection"),
                 ("remote os command", "Command Injection"), ("authentication", "Brute Force / Auth")]
RISK = {"0": "Informational", "1": "Low", "2": "Medium", "3": "High"}
FIELDS = ["alert", "risk", "cwe", "category", "method", "url", "param", "evidence"]


def category(name: str, cweid) -> str:
    try:
        if int(cweid) in CWE_MAP:
            return CWE_MAP[int(cweid)]
    except (TypeError, ValueError):
        pass
    low = (name or "").lower()
    return next((cat for kw, cat in NAME_KEYWORDS if kw in low), "Other")


def parse(report_json: str) -> list[dict]:
    with open(report_json, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for site in data.get("site", []):
        for alert in site.get("alerts", []):
            for inst in alert.get("instances") or [{}]:
                rows.append({"alert": alert.get("name") or alert.get("alert", ""),
                             "risk": RISK.get(str(alert.get("riskcode")), str(alert.get("riskcode"))),
                             "cwe": alert.get("cweid", ""),
                             "category": category(alert.get("name") or alert.get("alert"), alert.get("cweid")),
                             "method": inst.get("method", ""), "url": inst.get("uri", ""),
                             "param": inst.get("param", ""), "evidence": (inst.get("evidence") or "")[:120]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report_json")
    ap.add_argument("out_csv")
    args = ap.parse_args()
    rows = parse(args.report_json)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} ZAP alert instances -> {args.out_csv}")
    print("by risk    :", dict(Counter(r["risk"] for r in rows)))
    print("by category:", dict(Counter(r["category"] for r in rows)))


if __name__ == "__main__":
    main()

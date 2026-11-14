"""
compare_with_zap.py
Put the conventional scanner's findings (OWASP ZAP) side by side with the AI detectors'
results on the same lab session, and write a Markdown + CSV summary for the report.

  python compare_with_zap.py --zap-json lab_run/zap_report.json \
      --benign lab_run/benign_scored.summary.json --attack lab_run/attack_scored.summary.json \
      --out lab_run/comparison.md

What is compared (they measure different things, so they are reported side by side):
  * ZAP (active scanner): vulnerabilities it CONFIRMED in the application, by risk level
    and category -- "what is wrong with the app".
  * AI detectors (passive, on network flows): share of the scanner's attack traffic they
    flag (detection rate) and share of normal browsing they flag (false-positive rate)
    -- "would we notice someone attacking the app".
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from parse_zap_report import parse  # noqa: E402


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional 'tabulate' dependency."""
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zap-json", required=True)
    ap.add_argument("--benign", required=True, help="summary.json from score_flows.py --phase benign")
    ap.add_argument("--attack", required=True, help="summary.json from score_flows.py --phase attack")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    zap = parse(args.zap_json)
    ben = json.loads(Path(args.benign).read_text())
    att = json.loads(Path(args.attack).read_text())
    risk = Counter(r["risk"] for r in zap)
    cats = Counter(r["category"] for r in zap)
    unique_alert_types = len({r["alert"] for r in zap})

    rows = []
    for n, d in att["detectors"].items():
        b = ben["detectors"].get(n, {})
        rows.append({"Detector": config.DISPLAY.get(n, n),
                     "Detection rate on ZAP attack traffic": round(d["alert_rate"], 4),
                     "Attack flows flagged": f"{d['alerts']}/{att['n_flows']}",
                     "False-positive rate on benign browsing": round(b.get("alert_rate", float("nan")), 4),
                     "Benign flows flagged": f"{b.get('alerts', 0)}/{ben['n_flows']}"})
    table = pd.DataFrame(rows)
    table.to_csv(Path(args.out).with_suffix(".csv"), index=False)

    md = ["# Lab comparison: OWASP ZAP vs. AI-assisted detection", "",
          "## Conventional scanner (OWASP ZAP active scan)", "",
          f"- Alert instances: **{len(zap)}** ({unique_alert_types} distinct alert types)",
          "- By risk: " + ", ".join(f"{k}: {v}" for k, v in sorted(risk.items())),
          "- By category: " + ", ".join(f"{k}: {v}" for k, v in cats.most_common()), "",
          "## AI-assisted detection (flow-based, passive)", "",
          f"Benign browsing flows: {ben['n_flows']} | ZAP attack flows: {att['n_flows']}", "",
          md_table(table), "",
          "## Reading the comparison", "",
          "ZAP confirms *vulnerabilities* by actively exploiting the app; the AI detectors flag "
          "*attack traffic* passively from flow statistics. A high detection rate means the AI "
          "would notice a scanner-style attack in progress; ZAP's findings say which flaws exist."]
    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

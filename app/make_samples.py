"""
make_samples.py -- builds the UI's demo files in app/samples/ (run from the repo root):

    python app/make_samples.py

  cic_flows_sample.csv   flows from the CIC-IDS2017 *test* split (never seen in training)
  demo_traffic.jsonl     a stratified sample of the latest lab run (benign shoppers + ZAP scan),
                         or, before any lab run, HttpParamsDataset test payloads placed in
                         Juice-Shop-like requests (demo_traffic.csv with a label column)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

import config  # noqa: E402
import data as D  # noqa: E402

OUT = ROOT / "app" / "samples"
OUT.mkdir(parents=True, exist_ok=True)


def per_group(df, col, n):
    return pd.concat([g.sample(min(len(g), n), random_state=0) for _, g in df.groupby(col)])


def cic():
    df = D.load_http()
    X, labels, y = D.xy(df)
    _, te = D.split_indices(labels)
    t = df.iloc[te]
    s = pd.concat([t[t[config.LABEL_COL] == config.BENIGN].sample(300, random_state=0),
                   per_group(t[t[config.LABEL_COL] != config.BENIGN], config.LABEL_COL, 60)]).sample(frac=1, random_state=0)
    s[D.features() + [config.LABEL_COL]].to_csv(OUT / "cic_flows_sample.csv", index=False)
    print(f"cic_flows_sample.csv: {len(s)} flows {s[config.LABEL_COL].value_counts().to_dict()}")


def traffic():
    runs = sorted((config.LAB_DIR / "runs").glob("*/requests.jsonl"))
    if runs:
        from lab.compare import truth
        recs = [json.loads(line) for line in open(runs[-1], encoding="utf-8") if line.strip()]
        recs = [r for r in recs if r["phase"] in ("benign", "zap")]
        d = pd.DataFrame({"i": range(len(recs)), "t": [truth(r) for r in recs]})
        pick = per_group(d, "t", 250)
        sel = [recs[i] for i in sorted(pick["i"])]
        with open(OUT / "demo_traffic.jsonl", "w", encoding="utf-8") as f:
            for r in sel:
                f.write(json.dumps(r) + "\n")
        print(f"demo_traffic.jsonl: {len(sel)} requests from lab run {runs[-1].parent.name}")
        return
    from webreq.datasets import load_httpparams
    _, test = load_httpparams()
    s = per_group(test, "attack_type", 60)
    eps = ["/rest/products/search?q={}", "/api/Feedbacks/?comment={}", "/rest/track-order/{}", "/ftp/{}"]
    rows = [{"method": "GET", "url": "http://shop.local" + eps[i % len(eps)].format(quote(p, safe="")),
             "label": t} for i, (p, t) in enumerate(zip(s["payload"], s["attack_type"]))]
    pd.DataFrame(rows).to_csv(OUT / "demo_traffic.csv", index=False)
    print(f"demo_traffic.csv: {len(rows)} requests (no lab run yet)")


if __name__ == "__main__":
    cic()
    traffic()

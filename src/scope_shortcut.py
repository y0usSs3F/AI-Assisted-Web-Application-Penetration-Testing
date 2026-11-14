"""
scope_shortcut.py  --  E6: how much does a careless evaluation scope inflate results?

Many CIC-IDS2017 studies train on ALL Thursday-morning traffic. Because every web attack
targets port 80, the "Destination Port" feature (or simply the web-vs-non-web nature of a
flow) separates most benign traffic from attacks for free. This script quantifies that
shortcut on the same held-out protocol:
  1. a trivial rule "alarm if Destination Port == 80"            (no learning at all)
  2. Random Forest on all traffic, WITH Destination Port
  3. Random Forest on all traffic, WITHOUT Destination Port
  4. Random Forest on HTTP flows only (the scope used in this project; from train_eval.py)
False-positive rates in 1-3 are diluted by tens of thousands of easy non-web flows, which
is why this project evaluates on HTTP flows only.

Run after train_eval.py:   python scope_shortcut.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
import metrics as Mx
import models as M

PORT = "Destination Port"


def main():
    df = pd.read_csv(config.THURSDAY_ALL)
    labels = df[config.LABEL_COL].to_numpy()
    y = (labels != config.BENIGN).astype(int)
    feats = [c for c in df.columns if c != config.LABEL_COL]
    X = df[feats].astype(float)
    tr, te = train_test_split(np.arange(len(df)), test_size=config.TEST_SIZE,
                              random_state=config.RANDOM_STATE, stratify=labels)
    print(f"All Thursday-morning traffic: {len(df):,} flows ({y.mean():.2%} attacks); "
          f"benign web share = {np.mean(X[PORT].to_numpy()[y == 0] == config.HTTP_PORT):.1%}")

    res = {}
    port80 = (X[PORT].to_numpy() == config.HTTP_PORT).astype(int)
    res["port80_rule"] = Mx.binary_metrics(y[te], port80[te], port80[te])
    for tag, cols in [("rf_all_with_port", feats), ("rf_all_without_port", [c for c in feats if c != PORT])]:
        rf = M.build_models()["rf"]
        rf.fit(X.iloc[tr][cols], y[tr])
        pred = rf.predict(X.iloc[te][cols])
        res[tag] = Mx.binary_metrics(y[te], pred, M.attack_score(rf, X.iloc[te][cols]))
    http = json.loads((config.RESULTS_DIR / "metrics.json").read_text())["binary"]["rf"]
    res["rf_http_only"] = {k: http[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc",
                                                 "fpr", "fnr", "mcc", "tpr_at_1pct_fpr")}
    res["n_test_benign_all"] = int((y[te] == 0).sum())

    names = {"port80_rule": "Trivial rule: port == 80", "rf_all_with_port": "RF, all traffic, with port",
             "rf_all_without_port": "RF, all traffic, no port", "rf_http_only": "RF, HTTP flows only (ours)"}
    rows = []
    for k, label in names.items():
        m = res[k]
        rows.append({"Setting": label, "Accuracy": m["accuracy"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"], "FPR": m["fpr"],
                     "MCC": m["mcc"]})
        print(f"  {label:30s} acc {m['accuracy']:.4f}  P {m['precision']:.4f}  R {m['recall']:.4f}  "
              f"F1 {m['f1']:.4f}  AUC {m['roc_auc']:.4f}  FPR {m['fpr']:.4f}")
    (config.RESULTS_DIR / "scope_shortcut.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(rows).round(4).to_csv(config.TAB_DIR / "table_scope_shortcut.csv", index=False)
    print("Saved results/scope_shortcut.json")


if __name__ == "__main__":
    main()

"""
unseen_attacks.py  --  E3: detecting an attack TYPE the detector has never seen.

A supervised model can only learn the attacks present in its training data. To test
zero-day-style generalisation, each attack type A is held out in turn:
  * train each binary detector on benign flows + the OTHER two attack types (train split);
  * test on ALL flows of type A (never seen) plus the benign test flows.
The resulting "unseen" detection rate is compared with the "seen" detection rate of the
very same flows from the pooled out-of-fold cross-validation (where A WAS in training).

The Threshold IDS never learns from attacks, so it is unaffected by holding A out --
it is the natural reference point for generalisation to novel attacks.

Run after train_eval.py:   python unseen_attacks.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
import data as D
import models as M
from baseline import ThresholdIDS


def main():
    df = D.load_http()
    X, labels, y = D.xy(df)
    tr, te = D.split_indices(labels)
    cv = json.loads((config.RESULTS_DIR / "metrics.json").read_text())["cv"]["binary"]

    res, rows = {}, []
    for held_out in config.ATTACKS:
        train_idx = tr[labels[tr] != held_out]
        test_idx = np.concatenate([te[labels[te] == config.BENIGN], np.where(labels == held_out)[0]])
        lt, yt = labels[test_idx], y[test_idx]
        n_unseen = int((lt == held_out).sum())
        print(f"\n=== held-out attack type: {held_out} ({n_unseen} flows never seen in training) ===")
        res[held_out] = {"n_flows": n_unseen}

        detectors = dict(M.build_models())
        detectors[config.BASELINE] = ThresholdIDS()
        for name, det in detectors.items():
            if name == config.BASELINE:
                det.fit(X.iloc[train_idx], y[train_idx])
                score = det.decision_function(X.iloc[test_idx])
            else:
                M.fit(name, det, X.iloc[train_idx], y[train_idx])
                score = M.attack_score(det, X.iloc[test_idx])
            pred = det.predict(X.iloc[test_idx])
            r = {"unseen_recall": float(pred[lt == held_out].mean()),
                 "seen_recall_oof": cv[name]["oof_recall_by_attack"][held_out],
                 "fpr": float(pred[yt == 0].mean()),
                 "roc_auc_unseen": float(roc_auc_score(yt, score))}
            res[held_out][name] = r
            rows.append({"Held-out attack": held_out, "Detector": config.DISPLAY[name],
                         "Seen recall (CV)": r["seen_recall_oof"], "Unseen recall": r["unseen_recall"],
                         "Drop (pts)": 100 * (r["seen_recall_oof"] - r["unseen_recall"]),
                         "FPR": r["fpr"], "ROC-AUC (unseen)": r["roc_auc_unseen"]})
            print(f"  {config.DISPLAY[name]:34s} seen {r['seen_recall_oof']:.3f} -> unseen "
                  f"{r['unseen_recall']:.3f}   AUC {r['roc_auc_unseen']:.3f}  FPR {r['fpr']:.4f}")

    (config.RESULTS_DIR / "unseen_attacks.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(rows).round(4).to_csv(config.TAB_DIR / "table_unseen_attacks.csv", index=False)
    print("\nSaved results/unseen_attacks.json")


if __name__ == "__main__":
    main()

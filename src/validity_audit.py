"""
validity_audit.py  --  E7: what are the detectors actually detecting?

CIC-IDS2017's web-attack flows have two documented validity problems (Engelen et al.,
2021; Lanvin et al., 2022). This experiment measures both in our data and re-evaluates
every detector with them controlled.

  1. Payload-less artifact flows.  Most flows labelled as web attacks carry NO application
     payload (0 bytes in both directions): they are TCP connection fragments (mostly
     3-packet set-up/tear-down remnants) that the flow meter split off from the real attack
     connections, labelled "attack" only because of their time window and endpoints.
     An empty connection cannot reveal whether it belonged to a brute-force or an XSS run.
  2. A TCP-stack fingerprint.  The attacker's Linux machine opens connections with TCP
     initial window 29200, so Init_Win_bytes_forward (with the victim server's
     Init_Win_bytes_backward and min_seg_size_forward) identifies the attacking HOST
     rather than the attack BEHAVIOUR -- a shortcut that would not survive a new attacker.

Settings (stratified 5-fold CV, pooled out-of-fold predictions, all detectors):
  S0  all HTTP flows, all features                      (standard setup = E1/E2)
  S1  all HTTP flows, fingerprint features removed
  S2  payload-bearing flows only (client sent > 0 bytes), all features
  S3  payload-bearing flows only, fingerprint features removed   (most conservative)
Both binary detection and multi-class attack-type classification are re-run per setting.

Run after download_data.py:   python validity_audit.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config
import data as D
import metrics as Mx
import models as M
from baseline import ThresholdIDS

FINGERPRINT = ["Init_Win_bytes_forward", "Init_Win_bytes_backward", "min_seg_size_forward"]
PAYLOAD_COL = "Total Length of Fwd Packets"
ATTACKER_WINDOW = 29200
ALL = config.ML_MODELS + [config.BASELINE]


def artifact_statistics(X, L) -> dict:
    fwd = X[PAYLOAD_COL]
    bwd = X["Total Length of Bwd Packets"]
    win = X["Init_Win_bytes_forward"]
    stats = {}
    for c in config.CLASSES:
        m = L == c
        stats[c] = {
            "n": int(m.sum()),
            "share_no_payload_either_direction": float(((fwd[m] == 0) & (bwd[m] == 0)).mean()),
            "n_with_client_payload": int((fwd[m] > 0).sum()),
            "share_window_29200": float((win[m] == ATTACKER_WINDOW).mean()),
        }
    art = (L != config.BENIGN) & (fwd == 0)
    stats["artifact_attack_flows_total_fwd_packets"] = {
        str(int(k)): int(v) for k, v in X.loc[art, "Total Fwd Packets"].value_counts().head(5).items()}
    return stats


def cv_eval(X, L, y) -> tuple[dict, dict]:
    skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    pred = {n: np.zeros(len(y), dtype=int) for n in ALL}
    score = {n: np.zeros(len(y)) for n in ALL}
    mc = {n: np.empty(len(y), dtype=object) for n in config.ML_MODELS}
    for tr, te in skf.split(X, L):
        for name, mdl in M.build_models().items():
            M.fit(name, mdl, X.iloc[tr], y[tr])
            pred[name][te] = mdl.predict(X.iloc[te]); score[name][te] = M.attack_score(mdl, X.iloc[te])
        ids = ThresholdIDS().fit(X.iloc[tr], y[tr])
        pred[config.BASELINE][te] = ids.predict(X.iloc[te])
        score[config.BASELINE][te] = ids.decision_function(X.iloc[te])
        for name, mdl in M.build_models().items():
            M.fit(name, mdl, X.iloc[tr], L[tr])
            mc[name][te] = mdl.predict(X.iloc[te])
    binary = {n: {**Mx.binary_metrics(y, pred[n], score[n]),
                  "recall_by_attack": Mx.recall_by_attack(L, pred[n])} for n in ALL}
    multi = {}
    for n in config.ML_MODELS:
        r = Mx.multiclass_metrics(L, mc[n].astype(str))
        multi[n] = {"macro_f1": r["macro_f1"],
                    "f1_by_class": {c: r["report"][c]["f1-score"] for c in config.CLASSES},
                    "confusion": r["confusion"]}
    return binary, multi


def main():
    df = D.load_http()
    X, L, y = D.xy(df)
    stats = artifact_statistics(X, L)
    print("Artifact / fingerprint statistics:")
    for c in config.CLASSES:
        s = stats[c]
        print(f"  {c:14s} no payload: {s['share_no_payload_either_direction']:.1%}   "
              f"with payload: {s['n_with_client_payload']:5d}   TCP window 29200: {s['share_window_29200']:.1%}")

    payload = (X[PAYLOAD_COL] > 0).to_numpy()
    no_fp = [c for c in X.columns if c not in FINGERPRINT]
    settings = {
        "S0": ("All HTTP flows, all features", np.ones(len(y), bool), list(X.columns)),
        "S1": ("All HTTP flows, no TCP fingerprint", np.ones(len(y), bool), no_fp),
        "S2": ("Payload-bearing flows, all features", payload, list(X.columns)),
        "S3": ("Payload-bearing flows, no TCP fingerprint", payload, no_fp),
    }
    out = {"statistics": stats, "fingerprint_features": FINGERPRINT, "settings": {}}
    rows = []
    for key, (desc, mask, cols) in settings.items():
        t0 = time.time()
        Xs, Ls, ys = X[mask][cols], L[mask], y[mask]
        binary, multi = cv_eval(Xs, Ls, ys)
        counts = pd.Series(Ls).value_counts().to_dict()
        out["settings"][key] = {"description": desc, "n_flows": int(len(ys)), "n_features": len(cols),
                                "counts": counts, "binary": binary, "multiclass": multi}
        print(f"\n{key}  {desc}: {len(ys):,} flows, {int(ys.sum())} attacks {counts}  ({time.time()-t0:.0f}s)")
        for n in ALL:
            b = binary[n]; ra = b["recall_by_attack"]
            mf = multi[n]["macro_f1"] if n in multi else None
            rows.append({"Setting": key, "Description": desc, "Detector": config.DISPLAY[n],
                         "Attack flows": int(ys.sum()), "F1": b["f1"], "Precision": b["precision"],
                         "Recall": b["recall"], "ROC-AUC": b["roc_auc"], "FPR": b["fpr"],
                         "Recall BruteForce": ra.get("Brute Force"), "Recall XSS": ra.get("XSS"),
                         "Recall SQLi": ra.get("SQL Injection"), "Multi-class macro-F1": mf})
            print(f"  {config.DISPLAY[n]:34s} F1 {b['f1']:.3f}  P {b['precision']:.3f}  R {b['recall']:.3f}  "
                  f"AUC {b['roc_auc']:.4f}  FPR {b['fpr']:.4f}" + (f"  | type macro-F1 {mf:.3f}" if mf is not None else ""))

    (config.RESULTS_DIR / "validity_audit.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).round(4).to_csv(config.TAB_DIR / "table_validity_audit.csv", index=False)
    print("\nSaved results/validity_audit.json")


if __name__ == "__main__":
    main()

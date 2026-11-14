"""
sanity.py -- "are these accuracies real?" checks for the request-level results
(run from src/:  python -m webreq.sanity   or   python -m webreq.sanity --unified)

High scores can come from mistakes rather than from learning attacks. Each check below
targets one known trap:

  S1 Label-permutation test   Train on randomly shuffled labels. A sound pipeline must fall to
                              chance (ROC-AUC ~ 0.5). A high score would prove leakage, e.g.
                              test information reaching training through preprocessing.
  S2 Train/test near-duplicates
                              Requests that differ only in digits or letter case are counted
                              as near-duplicates. The share of test requests with a
                              near-duplicate in training is reported, and the model is
                              re-scored on the novel test requests only (memorisation check).
  S3 Bootstrap confidence intervals
                              95% CIs (1,000 resamples of the test set) for F1, precision,
                              recall and FPR: how much of a difference between models is noise.
  S4 Shortcut check           A model that sees only "boring" request properties (method, path
                              depth, lengths, number of parameters, no content) shows how much
                              of the score is available without inspecting what was sent.
Writes results/webreq/sanity[_unified].json and a table.
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

import config
from webreq import datasets, models as WM

OUT = config.WEB_RESULTS
RS = config.RANDOM_STATE
SHORTCUT_FEATURES = ["method_post", "method_other", "path_len", "path_depth", "n_params",
                     "query_len", "body_len", "max_value_len", "mean_value_len"]


def _norm(r) -> str:
    """Near-duplicate key: decoded text, lower-case, every digit run -> 0."""
    return re.sub(r"\d+", "0", r.to_text())


def _fpr(y, p):
    neg = y == 0
    return float((p[neg] == 1).mean()) if neg.any() else float("nan")


def bootstrap(y, pred, n=1000, seed=RS) -> dict:
    rng = np.random.default_rng(seed)
    stats = {"f1": [], "precision": [], "recall": [], "fpr": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        stats["f1"].append(f1_score(y[i], pred[i], zero_division=0))
        stats["precision"].append(precision_score(y[i], pred[i], zero_division=0))
        stats["recall"].append(recall_score(y[i], pred[i], zero_division=0))
        stats["fpr"].append(_fpr(y[i], pred[i]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}


def run(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, tag: str, strat=None, profile=True,
        names=("xgb", "char_svm", "hybrid")) -> dict:
    tr, te = train_test_split(np.arange(len(y)), test_size=config.TEST_SIZE, random_state=RS,
                              stratify=strat if strat is not None else y)
    reqs = list(df["request"])
    res = {"n_train": len(tr), "n_test": len(te)}

    print(f"\n=== S1 label-permutation test ({tag}) ===")
    rng = np.random.default_rng(RS)
    res["S1_permutation"] = {}
    for name in ("xgb", "char_svm"):
        yp = y[tr].copy()
        rng.shuffle(yp)
        m = WM.build(profile=profile, names=[name])[name].fit(X.iloc[tr], yp)
        auc = roc_auc_score(y[te], WM.score(m, X.iloc[te]))
        f1 = f1_score(y[te], m.predict(X.iloc[te]))
        res["S1_permutation"][name] = {"roc_auc": float(auc), "f1": float(f1),
                                       "f1_of_predict_all_attack": float(f1_score(y[te], np.ones(len(te), int)))}
        print(f"  {WM.DISPLAY[name]:38s} shuffled-label ROC-AUC {auc:.3f} (chance 0.5), F1 {f1:.3f}")

    print(f"\n=== S2 train/test near-duplicates ({tag}) ===")
    keys = np.array([_norm(r) for r in reqs])
    train_keys = set(keys[tr])
    dup = np.array([k in train_keys for k in keys[te]])
    res["S2_near_duplicates"] = {"test_share_with_near_duplicate_in_train": float(dup.mean()),
                                 "by_class": {str(c): float(dup[y[te] == c].mean()) for c in (0, 1)}}
    print(f"  {dup.mean():.1%} of test requests have a near-duplicate in training "
          f"(normal {res['S2_near_duplicates']['by_class']['0']:.1%}, anomalous {res['S2_near_duplicates']['by_class']['1']:.1%})")

    print(f"\n=== S3 bootstrap 95% CIs and scores on novel test requests ({tag}) ===")
    res["S3"] = {}
    rows = []
    for name, mdl in WM.build(profile=profile, names=list(names)).items():
        mdl.fit(X.iloc[tr], y[tr])
        pred = mdl.predict(X.iloc[te]).astype(int)
        nov = ~dup
        r = {"f1": float(f1_score(y[te], pred)), "ci": bootstrap(y[te], pred),
             "f1_novel_only": float(f1_score(y[te][nov], pred[nov])) if nov.any() else None,
             "fpr_novel_only": _fpr(y[te][nov], pred[nov]) if nov.any() else None,
             "n_novel": int(nov.sum())}
        res["S3"][name] = r
        rows.append({"Check": tag, "Model": WM.DISPLAY[name], "Test F1": r["f1"],
                     "F1 95% CI": f"{r['ci']['f1'][0]:.4f}-{r['ci']['f1'][1]:.4f}",
                     "FPR 95% CI": f"{r['ci']['fpr'][0]:.4f}-{r['ci']['fpr'][1]:.4f}",
                     "F1 on novel test requests": r["f1_novel_only"],
                     "Shuffled-label ROC-AUC": res["S1_permutation"].get(name, {}).get("roc_auc")})
        print(f"  {WM.DISPLAY[name]:38s} F1 {r['f1']:.4f} [{r['ci']['f1'][0]:.4f}, {r['ci']['f1'][1]:.4f}]  "
              f"novel-only F1 {r['f1_novel_only']:.4f} (n={r['n_novel']:,})")

    print(f"\n=== S4 shortcut check: structure only, no content ({tag}) ===")
    m = WM.build(tuned=False, features=SHORTCUT_FEATURES, names=["xgb"])["xgb"].fit(X.iloc[tr], y[tr])
    p = m.predict(X.iloc[te])
    res["S4_structure_only"] = {"features": SHORTCUT_FEATURES, "f1": float(f1_score(y[te], p)),
                                "roc_auc": float(roc_auc_score(y[te], WM.score(m, X.iloc[te])))}
    print(f"  XGBoost on {len(SHORTCUT_FEATURES)} structural features only: F1 {res['S4_structure_only']['f1']:.4f}  "
          f"ROC-AUC {res['S4_structure_only']['roc_auc']:.4f}")
    return res, rows


def main():
    unified = "--unified" in sys.argv
    if unified:
        from webreq import unified as U
        df = U.load()
        tag = "Unified CSIC 2010 + SR-BH 2020"
        strat = (df["source"] + "|" + df["label"].astype(str)).to_numpy()
    else:
        df = datasets.load_csic()
        tag = "CSIC 2010"
        strat = None
    X = WM.to_frame(list(df["request"]))
    y = df["label"].to_numpy()
    res, rows = run(df, X, y, tag, strat=strat)
    suffix = "_unified" if unified else ""
    (OUT / f"sanity{suffix}.json").write_text(json.dumps(res, indent=2, default=float))
    pd.DataFrame(rows).round(4).to_csv(OUT / f"table_sanity{suffix}.csv", index=False)
    print(f"\nWrote results/webreq/sanity{suffix}.json")


if __name__ == "__main__":
    main()

"""
train_eval.py
Experiments 1 and 2 on the Thursday HTTP flows of CIC-IDS2017.

  E1  Binary detection   (benign vs. web attack)
      5 AI/ML models + the conventional Threshold IDS
  E2  Multi-class        (BENIGN / Brute Force / XSS / SQL Injection)
      5 AI/ML models (a threshold IDS raises alarms but cannot name the attack type)

Each task is evaluated two ways:
  (a) a held-out, previously unseen 20% test split  -> tables, ROC/PR curves, confusion matrices
  (b) stratified 5-fold cross-validation              -> mean +/- std, plus pooled out-of-fold
      predictions so every flow -- including all 21 SQL-injection flows -- is tested once.
Also computes permutation feature importance for the Random Forest detector.

Run after download_data.py:   python train_eval.py
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

import config
import data as D
import metrics as Mx
import models as M
from baseline import ThresholdIDS

ALL_BIN = config.ML_MODELS + [config.BASELINE]


def _save_pred(fname, **arrays):
    np.savez_compressed(config.PRED_DIR / fname, **arrays)


# ------------------------------------------------------------------ held-out evaluation
def binary_holdout(X, labels, y, tr, te) -> dict:
    print("\n=== E1  binary detection (held-out test) ===")
    res = {}
    for name, mdl in M.build_models().items():
        t0 = time.time(); M.fit(name, mdl, X.iloc[tr], y[tr]); fit_s = time.time() - t0
        t1 = time.time()
        pred = mdl.predict(X.iloc[te]); score = M.attack_score(mdl, X.iloc[te])
        inf_ms = 1000 * (time.time() - t1) / len(te)
        m = Mx.binary_metrics(y[te], pred, score)
        m.update(recall_by_attack=Mx.recall_by_attack(labels[te], pred),
                 train_s=fit_s, infer_ms_per_flow=inf_ms)
        res[name] = m
        joblib.dump(mdl, config.MODELS_DIR / f"binary_{name}.joblib")
        _save_pred(f"binary_{name}.npz", y_true=y[te], y_pred=pred, y_score=score, labels=labels[te])
        print(f"  {config.DISPLAY[name]:34s} F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}  "
              f"FPR={m['fpr']:.4f}  FNR={m['fnr']:.4f}  TPR@1%={m['tpr_at_1pct_fpr']:.4f}  ({fit_s:.1f}s)")

    ids = ThresholdIDS()
    t0 = time.time(); ids.fit(X.iloc[tr], y[tr]); fit_s = time.time() - t0
    t1 = time.time()
    pred = ids.predict(X.iloc[te]); score = ids.decision_function(X.iloc[te])
    inf_ms = 1000 * (time.time() - t1) / len(te)
    m = Mx.binary_metrics(y[te], pred, score)
    m.update(recall_by_attack=Mx.recall_by_attack(labels[te], pred), train_s=fit_s,
             infer_ms_per_flow=inf_ms, alarm_threshold_m=int(ids.m_),
             benign_train_fpr=ids.benign_fpr_)
    res[config.BASELINE] = m
    joblib.dump(ids, config.MODELS_DIR / f"binary_{config.BASELINE}.joblib")
    _save_pred(f"binary_{config.BASELINE}.npz", y_true=y[te], y_pred=pred, y_score=score, labels=labels[te])
    print(f"  {config.DISPLAY[config.BASELINE]:34s} F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}  "
          f"FPR={m['fpr']:.4f}  FNR={m['fnr']:.4f}  TPR@1%={m['tpr_at_1pct_fpr']:.4f}  "
          f"(alarm if >= {ids.m_} metrics out of range)")
    return res


def multiclass_holdout(X, labels, tr, te) -> dict:
    print("\n=== E2  multi-class classification (held-out test) ===")
    res = {}
    for name, mdl in M.build_models().items():
        t0 = time.time(); M.fit(name, mdl, X.iloc[tr], labels[tr]); fit_s = time.time() - t0
        pred = mdl.predict(X.iloc[te])
        m = Mx.multiclass_metrics(labels[te], pred)
        m["train_s"] = fit_s
        res[name] = m
        joblib.dump(mdl, config.MODELS_DIR / f"mc_{name}.joblib")
        _save_pred(f"mc_{name}.npz", y_true=labels[te].astype(str), y_pred=pred.astype(str))
        f1s = "  ".join(f"{c[:5]}={m['report'][c]['f1-score']:.3f}" for c in config.CLASSES)
        print(f"  {config.DISPLAY[name]:34s} macroF1={m['macro_f1']:.4f}  [{f1s}]")
    return res


# ------------------------------------------------------------------ cross-validation
def cross_validate(X, labels, y) -> dict:
    print(f"\n=== {config.CV_FOLDS}-fold stratified cross-validation ===")
    skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_bin = {n: {"f1": [], "roc_auc": [], "recall": [], "fpr": []} for n in ALL_BIN}
    fold_mc = {n: {"macro_f1": [], "accuracy": []} for n in config.ML_MODELS}
    oof_bin = {n: np.zeros(len(y), dtype=int) for n in ALL_BIN}
    oof_mc = {n: np.empty(len(y), dtype=object) for n in config.ML_MODELS}

    for k, (tr, te) in enumerate(skf.split(X, labels), 1):
        t0 = time.time()
        for name, mdl in M.build_models().items():
            M.fit(name, mdl, X.iloc[tr], y[tr])
            pred = mdl.predict(X.iloc[te]); score = M.attack_score(mdl, X.iloc[te])
            m = Mx.binary_metrics(y[te], pred, score)
            for key in fold_bin[name]:
                fold_bin[name][key].append(m[key])
            oof_bin[name][te] = pred
        ids = ThresholdIDS().fit(X.iloc[tr], y[tr])
        pred = ids.predict(X.iloc[te]); m = Mx.binary_metrics(y[te], pred, ids.decision_function(X.iloc[te]))
        for key in fold_bin[config.BASELINE]:
            fold_bin[config.BASELINE][key].append(m[key])
        oof_bin[config.BASELINE][te] = pred
        for name, mdl in M.build_models().items():
            M.fit(name, mdl, X.iloc[tr], labels[tr])
            pred = mdl.predict(X.iloc[te])
            mm = Mx.multiclass_metrics(labels[te], pred)
            fold_mc[name]["macro_f1"].append(mm["macro_f1"]); fold_mc[name]["accuracy"].append(mm["accuracy"])
            oof_mc[name][te] = pred
        print(f"  fold {k}/{config.CV_FOLDS} done ({time.time() - t0:.0f}s)")

    summary = {"binary": {}, "multiclass": {}}
    for n in ALL_BIN:
        summary["binary"][n] = {
            **{f"{k}_mean": float(np.mean(v)) for k, v in fold_bin[n].items()},
            **{f"{k}_std": float(np.std(v)) for k, v in fold_bin[n].items()},
            "folds": fold_bin[n],
            "oof_recall_by_attack": Mx.recall_by_attack(labels, oof_bin[n]),
            "oof_fpr": float(oof_bin[n][y == 0].mean()),
        }
    for n in config.ML_MODELS:
        rep = Mx.multiclass_metrics(labels, oof_mc[n].astype(str))
        summary["multiclass"][n] = {
            **{f"{k}_mean": float(np.mean(v)) for k, v in fold_mc[n].items()},
            **{f"{k}_std": float(np.std(v)) for k, v in fold_mc[n].items()},
            "folds": fold_mc[n],
            "oof_report": rep["report"], "oof_confusion": rep["confusion"],
        }
    for n in ALL_BIN:
        s = summary["binary"][n]
        print(f"  binary  {config.DISPLAY[n]:34s} F1 {s['f1_mean']:.4f} +/- {s['f1_std']:.4f}  "
              f"OOF recall {s['oof_recall_by_attack']}")
    for n in config.ML_MODELS:
        s = summary["multiclass"][n]
        print(f"  multi   {config.DISPLAY[n]:34s} macro-F1 {s['macro_f1_mean']:.4f} +/- {s['macro_f1_std']:.4f}")
    return summary


# ------------------------------------------------------------------ explanation
def feature_importance(X, y, te) -> dict:
    print("\n=== permutation feature importance (Random Forest, binary) ===")
    rf = joblib.load(config.MODELS_DIR / "binary_rf.joblib")
    pi = permutation_importance(rf, X.iloc[te], y[te], scoring="average_precision",
                                n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=1)
    order = np.argsort(pi.importances_mean)[::-1]
    names = list(X.columns)
    top = [(names[i], float(pi.importances_mean[i]), float(pi.importances_std[i])) for i in order[:20]]
    for n, v, s in top[:10]:
        print(f"  {n:28s} {v:.4f} +/- {s:.4f}")
    gini = sorted(zip(names, rf.feature_importances_), key=lambda t: -t[1])[:20]
    return {"permutation": top, "impurity": [(n, float(v)) for n, v in gini]}


# ------------------------------------------------------------------ tables
def write_tables(b, mc, cv):
    rows = []
    for n in ALL_BIN:
        m = b[n]; ra = m["recall_by_attack"]
        rows.append({"Detector": config.DISPLAY[n], "Accuracy": m["accuracy"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"],
                     "MCC": m["mcc"], "FPR": m["fpr"], "FNR": m["fnr"], "TPR@1%FPR": m["tpr_at_1pct_fpr"],
                     "Recall BruteForce": ra.get("Brute Force"), "Recall XSS": ra.get("XSS"),
                     "Recall SQLi": ra.get("SQL Injection"), "Train (s)": m["train_s"],
                     "Inference (ms/flow)": m["infer_ms_per_flow"]})
    pd.DataFrame(rows).round(4).to_csv(config.TAB_DIR / "table_binary.csv", index=False)

    rows = []
    for n in config.ML_MODELS:
        r = mc[n]["report"]
        rows.append({"Model": config.DISPLAY[n], "Accuracy": mc[n]["accuracy"], "Macro-F1": mc[n]["macro_f1"],
                     "Weighted-F1": mc[n]["weighted_f1"],
                     **{f"F1 {c}": r[c]["f1-score"] for c in config.CLASSES}})
    pd.DataFrame(rows).round(4).to_csv(config.TAB_DIR / "table_multiclass.csv", index=False)

    rows = []
    for n in ALL_BIN:
        s = cv["binary"][n]; ra = s["oof_recall_by_attack"]
        row = {"Detector": config.DISPLAY[n],
               "Binary F1 (mean)": s["f1_mean"], "Binary F1 (std)": s["f1_std"],
               "ROC-AUC (mean)": s["roc_auc_mean"], "FPR (mean)": s["fpr_mean"],
               "OOF recall BruteForce": ra.get("Brute Force"), "OOF recall XSS": ra.get("XSS"),
               "OOF recall SQLi": ra.get("SQL Injection")}
        if n in cv["multiclass"]:
            sm = cv["multiclass"][n]
            row.update({"Multi-class macro-F1 (mean)": sm["macro_f1_mean"],
                        "Multi-class macro-F1 (std)": sm["macro_f1_std"]})
        rows.append(row)
    pd.DataFrame(rows).round(4).to_csv(config.TAB_DIR / "table_cv.csv", index=False)

    best = max(config.ML_MODELS, key=lambda n: cv["multiclass"][n]["macro_f1_mean"])
    r = cv["multiclass"][best]["oof_report"]
    pd.DataFrame([{"Class": c, "Precision": r[c]["precision"], "Recall": r[c]["recall"],
                   "F1": r[c]["f1-score"], "Support": int(r[c]["support"])} for c in config.CLASSES]
                 ).round(4).to_csv(config.TAB_DIR / "table_perclass_oof_best.csv", index=False)
    return best


def main():
    t_start = time.time()
    df = D.load_http()
    X, labels, y = D.xy(df)
    tr, te = D.split_indices(labels)
    print(f"Thursday HTTP flows: {len(df):,} ({y.mean():.1%} attacks), {X.shape[1]} features; "
          f"train {len(tr):,} / test {len(te):,}")
    split_info = {"n_total": len(df), "n_train": len(tr), "n_test": len(te),
                  "test_counts": pd.Series(labels[te]).value_counts().to_dict(),
                  "train_counts": pd.Series(labels[tr]).value_counts().to_dict()}

    b = binary_holdout(X, labels, y, tr, te)
    mc = multiclass_holdout(X, labels, tr, te)
    cv = cross_validate(X, labels, y)
    fi = feature_importance(X, y, te)
    best = write_tables(b, mc, cv)

    out = {"split": split_info, "binary": b, "multiclass": mc, "cv": cv,
           "feature_importance": fi, "best_multiclass_model": best}
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved results/metrics.json and tables. Total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

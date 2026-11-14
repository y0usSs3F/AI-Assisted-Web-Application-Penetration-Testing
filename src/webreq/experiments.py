"""
experiments.py -- request-level experiments (run from src/:  python -m webreq.experiments)

  R1  Detection on HTTP CSIC 2010 (normal vs. anomalous request): four AI/ML models and the
      conventional signature baseline, on a held-out 20% test split and with stratified
      5-fold cross-validation, after removing duplicate requests. Feature models use the
      engineered features and the learned application profile, with the regularisation
      settings chosen in R5 (webreq/tuning.py, inner CV on the training split).
  R2  Generalisation to previously unseen data from another source: CSIC-trained detectors
      scored on the HttpParamsDataset test payloads (sqlmap, xssya, Vega and FuzzDB attacks).
  R3  Attack-type classification (norm / sqli / xss / cmdi / path-traversal) on the official
      HttpParamsDataset train/test split, compared with the signature rules' categories.
  R4  Which features matter: XGBoost gain importance, aggregated by feature group.
Also trains the prototype's final models (all labelled data) for the analyzer and the UI.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

import config
import metrics as Mx
from webreq import datasets, features as F, models as WM
from webreq.signatures import SignatureDetector

OUT = config.WEB_RESULTS
TYPES = ["norm", "sqli", "xss", "path-traversal", "cmdi"]


def _dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=float))


def _signature_category(req) -> str:
    from webreq.signatures import inspect, _majority
    return _majority(inspect(req)) or "none"


# ------------------------------------------------------------------ R1
def r1_detection(df: pd.DataFrame, X: pd.DataFrame) -> dict:
    y = df["label"].to_numpy()
    reqs = list(df["request"])
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=config.TEST_SIZE, stratify=y,
                              random_state=config.RANDOM_STATE)
    res = {"n_requests": len(df), "n_anomalous": int(y.sum()), "n_train": len(tr), "n_test": len(te),
           "holdout": {}, "cv": {}}
    print(f"\n=== R1 detection on CSIC 2010: {len(tr):,} train / {len(te):,} test ===")
    sig_pred, sig_cat, _ = SignatureDetector().detect(reqs)
    sig_pred = np.asarray(sig_pred)
    for name, mdl in WM.build(profile=True).items():
        t0 = time.time(); mdl.fit(X.iloc[tr], y[tr]); fit_s = time.time() - t0
        t1 = time.time(); pred = mdl.predict(X.iloc[te]); s = WM.score(mdl, X.iloc[te])
        inf_ms = 1000 * (time.time() - t1) / len(te)
        m = Mx.binary_metrics(y[te], pred, s) | {"train_s": fit_s, "infer_ms": inf_ms}
        res["holdout"][name] = m
        np.savez_compressed(config.PRED_DIR / f"webreq_r1_{name}.npz", y=y[te], score=s, pred=pred, idx=te)
        print(f"  {WM.DISPLAY[name]:36s} F1 {m['f1']:.4f}  P {m['precision']:.4f}  R {m['recall']:.4f}  "
              f"FPR {m['fpr']:.4f}  AUC {m['roc_auc']:.4f}")
        if name == "xgb":
            joblib.dump(mdl, config.MODELS_DIR / "webreq_eval_xgb.joblib")
    m = Mx.binary_metrics(y[te], sig_pred[te], sig_pred[te].astype(float))
    res["holdout"]["signature"] = m
    print(f"  {WM.DISPLAY['signature']:36s} F1 {m['f1']:.4f}  P {m['precision']:.4f}  R {m['recall']:.4f}  "
          f"FPR {m['fpr']:.4f}")

    # error analysis: what kind of anomalies does each detector miss?
    cats = np.array([c or "none" for c in sig_cat])
    res["anomaly_mix"] = pd.Series(cats[y == 1]).value_counts().to_dict()
    res["miss_by_signature_category"] = {}
    for name in ["xgb", "char_svm", "hybrid"]:
        p = np.load(config.PRED_DIR / f"webreq_r1_{name}.npz")
        fn = te[(y[te] == 1) & (p["pred"] == 0)]
        res["miss_by_signature_category"][name] = pd.Series(cats[fn]).value_counts().to_dict()
    res["fp_examples"] = []
    p = np.load(config.PRED_DIR / "webreq_r1_xgb.npz")
    for i in te[(y[te] == 0) & (p["pred"] == 1)][:5]:
        r = reqs[i]
        res["fp_examples"].append(f"{r.method} {r.path}?{r.query[:100]} {r.body[:100]}")
    res["fn_examples"] = []
    for i in te[(y[te] == 1) & (p["pred"] == 0)][:8]:
        r = reqs[i]
        res["fn_examples"].append(f"{r.method} {r.path}?{r.query[:100]} {r.body[:100]}")

    print("  5-fold cross-validation ...")
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    folds = {n: [] for n in WM.NAMES + ["signature"]}
    for k, (a, b) in enumerate(skf.split(idx, y)):
        for name, mdl in WM.build(profile=True).items():
            mdl.fit(X.iloc[a], y[a])
            folds[name].append(Mx.binary_metrics(y[b], mdl.predict(X.iloc[b]), WM.score(mdl, X.iloc[b])))
        folds["signature"].append(Mx.binary_metrics(y[b], sig_pred[b], sig_pred[b].astype(float)))
    for name, ms in folds.items():
        res["cv"][name] = {f"{k}_mean": float(np.mean([m[k] for m in ms])) for k in ("f1", "precision", "recall", "fpr", "roc_auc")}
        res["cv"][name] |= {f"{k}_std": float(np.std([m[k] for m in ms])) for k in ("f1", "precision", "recall", "fpr")}
        print(f"  CV {WM.DISPLAY[name]:36s} F1 {res['cv'][name]['f1_mean']:.4f} +/- {res['cv'][name]['f1_std']:.4f}")
    return res


# ------------------------------------------------------------------ R2
def r2_generalisation(df, X, ptest) -> dict:
    print("\n=== R2 CSIC-trained detectors on unseen HttpParams payloads ===")
    y = df["label"].to_numpy()
    reqs = [F.payload_request(v) for v in ptest["payload"]]
    Xp = WM.to_frame(reqs)
    truth = ptest["attack_type"].to_numpy()
    res = {}
    for name, mdl in WM.build(profile=False).items():      # profiles are app-specific
        mdl.fit(X, y)
        pred = mdl.predict(Xp)
        res[name] = {t: float(pred[truth == t].mean()) for t in TYPES}
        res[name]["attack_recall"] = float(pred[truth != "norm"].mean())
        print(f"  {WM.DISPLAY[name]:36s} " + "  ".join(f"{t}={res[name][t]:.3f}" for t in TYPES))
    sp, _, _ = SignatureDetector().detect(reqs)
    sp = np.asarray(sp)
    res["signature"] = {t: float(sp[truth == t].mean()) for t in TYPES} | {"attack_recall": float(sp[truth != "norm"].mean())}
    print(f"  {WM.DISPLAY['signature']:36s} " + "  ".join(f"{t}={res['signature'][t]:.3f}" for t in TYPES))
    res["note"] = "values are alert rates: for 'norm' this is the false-positive rate, otherwise recall"
    return res


# ------------------------------------------------------------------ R3
def r3_types(ptrain, ptest) -> dict:
    print("\n=== R3 attack-type classification on HttpParamsDataset ===")
    Xa = WM.to_frame([F.payload_request(v) for v in ptrain["payload"]])
    Xb = WM.to_frame([F.payload_request(v) for v in ptest["payload"]])
    ya, yb = ptrain["attack_type"].to_numpy(), ptest["attack_type"].to_numpy()
    res = {"n_train": len(ya), "n_test": len(yb), "train_counts": pd.Series(ya).value_counts().to_dict(),
           "test_counts": pd.Series(yb).value_counts().to_dict(), "models": {}}

    def rep(name, pred):
        cr = classification_report(yb, pred, labels=TYPES, output_dict=True, zero_division=0)
        r = {"accuracy": accuracy_score(yb, pred), "macro_f1": f1_score(yb, pred, labels=TYPES, average="macro"),
             "per_class_f1": {t: cr[t]["f1-score"] for t in TYPES},
             "per_class_recall": {t: cr[t]["recall"] for t in TYPES},
             "confusion": confusion_matrix(yb, pred, labels=TYPES).tolist()}
        res["models"][name] = r
        print(f"  {WM.DISPLAY[name]:36s} acc {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f}  "
              + "  ".join(f"{t}={r['per_class_f1'][t]:.3f}" for t in TYPES))

    for name, mdl in WM.build(profile=False).items():
        mdl.fit(Xa, ya)
        rep(name, mdl.predict(Xb))
    sig = SignatureDetector().classify_values(list(ptest["payload"]))
    sig = np.array(["norm" if s == "other" else s for s in sig])
    rep("signature", sig)
    return res


# ------------------------------------------------------------------ R4
def r4_importance(df, X) -> dict:
    print("\n=== R4 feature importance (XGBoost gain, CSIC 2010) ===")
    from webreq.engineering import PROFILE_FEATURES
    mdl = WM.build(profile=True)["xgb"].fit(X, df["label"].to_numpy())
    booster = mdl.steps[-1][1].model_.get_booster()
    gain = booster.get_score(importance_type="gain")
    names = F.FEATURES + PROFILE_FEATURES
    imp = {names[int(k[1:])]: v for k, v in gain.items()}
    tot = sum(imp.values())
    imp = {k: v / tot for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}
    group_of = {**F.FEATURE_GROUP, **{p: "profile (learned)" for p in PROFILE_FEATURES}}
    groups, engineered = {}, 0.0
    for k, v in imp.items():
        groups[group_of[k]] = groups.get(group_of[k], 0) + v
        engineered += v if k in F.ENGINEERED else 0
    for k, v in list(imp.items())[:10]:
        print(f"  {k:22s} {v:.3f}")
    print("  by group:", {k: round(v, 3) for k, v in groups.items()})
    print(f"  engineered content features: {engineered:.3f} of total gain")
    return {"features": imp, "groups": groups, "engineered_share": engineered}


# ------------------------------------------------------------------ final prototype models
def train_prototype(df=None, X=None, ptrain=None, ptest=None):
    """Prototype models.

    Payload models (the prototype's detector + classifier): calibrated char n-gram SVM and
    XGBoost, trained on all HttpParamsDataset payloads. Request models trained on all of
    CSIC 2010 are saved too, to measure their domain shift in the lab.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from webreq.analyzer import build_bundle
    print("\n=== training the prototype's final models on all labelled data ===")
    if ptrain is None:
        ptrain, ptest = datasets.load_httpparams()
    pall = pd.concat([ptrain, ptest], ignore_index=True)
    Xp = WM.to_frame([F.payload_request(v) for v in pall["payload"]])
    yp = pall["attack_type"].to_numpy()
    nlp = WM.build(profile=False)["char_svm"]
    nlp.steps[-1] = ("clf", CalibratedClassifierCV(nlp.steps[-1][1], method="sigmoid", cv=3))
    nlp.fit(Xp, yp)
    xgb = WM.build(profile=False)["xgb"].fit(Xp, yp)
    build_bundle(nlp, xgb, Xp[yp == "norm"])
    print(f"  payload models trained on {len(pall):,} payloads -> models/webreq_prototype.joblib")
    if df is None:
        df = datasets.load_csic()
        X = WM.to_frame(list(df["request"]))
    for name in ("xgb", "char_svm"):
        joblib.dump(WM.build(profile=False)[name].fit(X, df["label"].to_numpy()),
                    config.MODELS_DIR / f"webreq_csic_{name}.joblib")
    print(f"  CSIC request models trained on {len(df):,} requests -> models/webreq_csic_*.joblib")


def write_tables(r1, r2, r3):
    rows = []
    for n in WM.NAMES + ["signature"]:
        h, c = r1["holdout"][n], r1["cv"][n]
        rows.append({"Detector": WM.DISPLAY[n], "Precision": h["precision"], "Recall": h["recall"],
                     "F1": h["f1"], "ROC-AUC": h["roc_auc"], "FPR": h["fpr"],
                     "CV F1": c["f1_mean"], "CV F1 std": c.get("f1_std", 0),
                     "Train (s)": h.get("train_s"), "Inference (ms/req)": h.get("infer_ms")})
    pd.DataFrame(rows).round(4).to_csv(OUT / "table_r1_csic_detection.csv", index=False)
    pd.DataFrame([{"Detector": WM.DISPLAY[n], **r2[n]} for n in WM.NAMES + ["signature"]]).round(4) \
        .to_csv(OUT / "table_r2_generalisation.csv", index=False)
    pd.DataFrame([{"Model": WM.DISPLAY[n], "Accuracy": r3["models"][n]["accuracy"],
                   "Macro-F1": r3["models"][n]["macro_f1"],
                   **{f"F1 {t}": r3["models"][n]["per_class_f1"][t] for t in TYPES}}
                  for n in WM.NAMES + ["signature"]]).round(4) \
        .to_csv(OUT / "table_r3_attack_types.csv", index=False)


def main():
    datasets.download()
    df = datasets.load_csic()
    print(f"CSIC 2010: {df.attrs['n_raw']:,} requests -> {len(df):,} unique "
          f"({int(df.label.sum()):,} anomalous)")
    X = WM.to_frame(list(df["request"]))
    ptrain, ptest = datasets.load_httpparams()
    r1 = r1_detection(df, X)
    r1["n_raw"] = df.attrs["n_raw"]
    _dump("r1_csic_detection.json", r1)
    r2 = r2_generalisation(df, X, ptest); _dump("r2_generalisation.json", r2)
    r3 = r3_types(ptrain, ptest); _dump("r3_attack_types.json", r3)
    r4 = r4_importance(df, X); _dump("r4_feature_importance.json", r4)
    write_tables(r1, r2, r3)
    train_prototype(df, X, ptrain, ptest)
    print("\nDone. Results in results/webreq/")


if __name__ == "__main__":
    main()

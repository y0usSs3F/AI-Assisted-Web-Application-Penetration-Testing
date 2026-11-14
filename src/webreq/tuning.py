"""
tuning.py -- R5: preprocessing, feature engineering and regularisation study
(run from src/ before the main experiments:  python -m webreq.tuning)

Everything here uses only the CSIC 2010 *training* split (the same 80/20 split as R1), so the
held-out test set stays untouched until the final evaluation in R1.

  1. Exploratory data analysis: class balance, duplicates, request size by class, per-feature
     separation (mutual information with the label), constant and redundant feature pairs.
  2. Feature-set ablation (5-fold CV): base features -> + engineered features -> + learned
     application profile, for LR and XGBoost, plus char n-grams and the hybrid model.
  3. Regularisation / hyperparameter tuning with inner 3-fold CV (F1):
       LR      penalty L2 / elastic-net / L1, strength C, optional correlation filter
       RF      depth, leaf size, features per split
       XGBoost depth, min child weight, L1/L2 leaf penalties, shrinkage, row/column subsampling
       SVMs    C
     The default and the tuned model are then fitted on the full training split; the train
     F1 vs. held-out F1 gap shows how much each model overfits.
  4. Validation curves (train vs. CV F1) for LR (C), char-SVM (C) and XGBoost (max depth).
Writes results/webreq/r5_preprocessing.json, tuned_params.json and tables.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate,
                                     train_test_split, validation_curve)

import config
import metrics as Mx
from webreq import datasets, features as F, models as WM
from webreq.engineering import PROFILE_FEATURES, CorrelationFilter, ProfileFeatures

OUT = config.WEB_RESULTS
RS = config.RANDOM_STATE


def split(y):
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=config.TEST_SIZE, stratify=y, random_state=RS)


# ------------------------------------------------------------------ 1. EDA
def eda(df, X, y, tr, ptrain, ptest) -> dict:
    print("\n=== R5.1 exploratory data analysis ===")
    Xt, yt = X.iloc[tr], y[tr]
    prof = ProfileFeatures().fit(Xt[["ctx"]], yt)
    P = pd.DataFrame(prof.transform(Xt[["ctx"]]), columns=PROFILE_FEATURES, index=Xt.index)
    A = pd.concat([Xt[F.FEATURES], P], axis=1).astype(float)
    mi = mutual_info_classif(A, yt, random_state=RS)
    C = A.corr().abs()
    pairs = [(a, b, float(C.loc[a, b])) for i, a in enumerate(A.columns) for b in A.columns[i + 1:]
             if C.loc[a, b] > 0.95]
    groups = {**F.FEATURE_GROUP, **{p: "profile (learned)" for p in PROFILE_FEATURES}}
    tab = pd.DataFrame({"feature": A.columns, "group": [groups[c] for c in A.columns],
                        "engineered": [c in F.ENGINEERED or c in PROFILE_FEATURES for c in A.columns],
                        "mean normal": A[yt == 0].mean().values, "mean anomalous": A[yt == 1].mean().values,
                        "mutual information": mi, "constant": (A.std() == 0).values}) \
        .sort_values("mutual information", ascending=False)
    tab.round(4).to_csv(OUT / "table_r5_eda_features.csv", index=False)
    size = lambda m: {c: float(X.loc[m, c].median()) for c in ("path_len", "query_len", "body_len", "n_params")}  # noqa: E731
    res = {
        "csic": {"raw_requests": int(df.attrs["n_raw"]), "unique_requests": len(df),
                 "duplicates_removed": int(df.attrs["n_raw"] - len(df) - df.attrs.get("n_conflicting", 0)),
                 "conflicting_labels_removed": int(df.attrs.get("n_conflicting", 0)),
                 "class_counts": {"normal": int((y == 0).sum()), "anomalous": int((y == 1).sum())},
                 "anomalous_share": float(y.mean()),
                 "methods": df["request"].map(lambda r: r.method).value_counts().to_dict(),
                 "median_size_normal": size(y == 0), "median_size_anomalous": size(y == 1)},
        "httpparams": {"train_counts": ptrain["attack_type"].value_counts().to_dict(),
                       "test_counts": ptest["attack_type"].value_counts().to_dict(),
                       "median_payload_len": pd.concat([ptrain, ptest]).assign(n=lambda d: d.payload.str.len())
                       .groupby("attack_type")["n"].median().to_dict()},
        "top_mutual_information": tab.head(12)[["feature", "mutual information"]].values.tolist(),
        "engineered_in_top12": int(tab.head(12)["engineered"].sum()),
        "constant_features": tab.loc[tab["constant"], "feature"].tolist(),
        "redundant_pairs_r_gt_0.95": pairs,
    }
    print(f"  CSIC {res['csic']['unique_requests']:,} unique requests ({res['csic']['anomalous_share']:.1%} anomalous); "
          f"{len(pairs)} redundant pairs; constant: {res['constant_features']}")
    print("  top MI:", ", ".join(f"{f} {v:.3f}" for f, v in res["top_mutual_information"][:8]))
    return res


# ------------------------------------------------------------------ 2. ablation
def _cv(model, X, y, n_jobs):
    skf = StratifiedKFold(5, shuffle=True, random_state=RS)
    r = cross_validate(model, X, y, cv=skf, n_jobs=n_jobs,
                       scoring={"f1": "f1", "precision": "precision", "recall": "recall", "auc": "roc_auc"})
    return {k: (float(r[f"test_{k}"].mean()), float(r[f"test_{k}"].std())) for k in ("f1", "precision", "recall", "auc")}


def ablation(X, y, tr) -> list:
    print("\n=== R5.2 feature-set ablation (5-fold CV on the training split, default hyperparameters) ===")
    Xt, yt = X.iloc[tr], y[tr]
    configs = [
        ("LR", "base features (35)", dict(features=F.BASE_FEATURES), "lr", 5),
        ("LR", "+ engineered (44)", dict(), "lr", 5),
        ("LR", "+ engineered + learned profile (49)", dict(profile=True), "lr", 5),
        ("XGBoost", "base features (35)", dict(features=F.BASE_FEATURES), "xgb", 1),
        ("XGBoost", "+ engineered (44)", dict(), "xgb", 1),
        ("XGBoost", "+ engineered + learned profile (49)", dict(profile=True), "xgb", 1),
        ("Char-SVM", "char 1-4-gram TF-IDF", dict(), "char_svm", 5),
        ("Hybrid", "char n-grams + features (44)", dict(), "hybrid", 5),
        ("Hybrid", "char n-grams + features + profile (49)", dict(profile=True), "hybrid", 5),
    ]
    rows = []
    for model, fs, kw, name, nj in configs:
        t0 = time.time()
        r = _cv(WM.build(tuned=False, names=[name], **kw)[name], Xt, yt, nj)
        rows.append({"Model": model, "Feature set": fs, "CV F1": r["f1"][0], "CV F1 std": r["f1"][1],
                     "Precision": r["precision"][0], "Recall": r["recall"][0], "ROC-AUC": r["auc"][0]})
        print(f"  {model:9s} {fs:42s} F1 {r['f1'][0]:.4f} +/- {r['f1'][1]:.4f}  "
              f"P {r['precision'][0]:.4f}  R {r['recall'][0]:.4f}  ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).round(4).to_csv(OUT / "table_r5_ablation.csv", index=False)
    return rows


# ------------------------------------------------------------------ 3. tuning
SPACES = {
    "lr": ("grid", [{"sel": ["passthrough", CorrelationFilter(0.95)], "clf__C": [0.01, 0.1, 1, 10, 100],
                     "clf__l1_ratio": [0.0], "clf__solver": ["lbfgs"]},
                    {"sel": ["passthrough", CorrelationFilter(0.95)], "clf__C": [0.01, 0.1, 1, 10, 100],
                     "clf__l1_ratio": [0.5, 1.0], "clf__solver": ["saga"]}]),
    "rf": ("random", {"clf__max_depth": [None, 15, 30], "clf__min_samples_leaf": [1, 2, 5],
                      "clf__max_features": ["sqrt", 0.3, 0.6]}, 10),
    "xgb": ("random", {"clf__n_estimators": [200, 400, 700], "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
                       "clf__max_depth": [3, 4, 6, 8, 10], "clf__min_child_weight": [1, 3, 5, 10],
                       "clf__subsample": [0.7, 0.85, 1.0], "clf__colsample_bytree": [0.6, 0.8, 1.0],
                       "clf__reg_lambda": [0.1, 1, 5, 10], "clf__reg_alpha": [0, 0.1, 1],
                       "clf__gamma": [0, 0.5, 1]}, 30),
    "char_svm": ("grid", {"clf__C": [0.1, 0.25, 0.5, 1, 2]}),
    "hybrid": ("grid", {"clf__C": [0.05, 0.1, 0.25, 0.5, 1]}),
}
PROFILE_MODELS = {"lr", "rf", "xgb", "hybrid"}       # app-specific profile used on CSIC


def _jsonable(v):
    if isinstance(v, CorrelationFilter):
        return "CorrelationFilter(0.95)"
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def tune(X, y, tr, te) -> tuple[dict, list]:
    print("\n=== R5.3 regularisation / hyperparameter tuning (inner 3-fold CV on the training split) ===")
    Xt, yt, Xe, ye = X.iloc[tr], y[tr], X.iloc[te], y[te]
    inner = StratifiedKFold(3, shuffle=True, random_state=RS)
    best, rows, searches = {}, [], {}
    for name, space in SPACES.items():
        t0 = time.time()
        prof = name in PROFILE_MODELS
        est = WM.build(profile=prof, tuned=False, names=[name])[name]
        nj = 1 if name in ("xgb", "rf") else 5
        if space[0] == "grid":
            s = GridSearchCV(est, space[1], scoring="f1", cv=inner, n_jobs=nj, return_train_score=True)
        else:
            s = RandomizedSearchCV(est, space[1], n_iter=space[2], scoring="f1", cv=inner, n_jobs=nj,
                                   random_state=RS, return_train_score=True)
        s.fit(Xt, yt)
        searches[name] = s
        best[name] = {k: _jsonable(v) for k, v in s.best_params_.items()}
        for label, mdl in (("default", WM.build(profile=prof, tuned=False, names=[name])[name]),
                           ("tuned", s.best_estimator_)):
            if label == "default":
                mdl.fit(Xt, yt)
            tr_f1 = f1_score(yt, mdl.predict(Xt))
            m = Mx.binary_metrics(ye, mdl.predict(Xe), WM.score(mdl, Xe))
            rows.append({"Model": WM.DISPLAY[name], "Setting": label, "Inner-CV F1": s.best_score_ if label == "tuned"
                         else np.nan, "Train F1": tr_f1, "Test F1": m["f1"], "Test FPR": m["fpr"],
                         "Test recall": m["recall"], "Overfit gap (train-test F1)": tr_f1 - m["f1"],
                         "Parameters": json.dumps(best[name]) if label == "tuned" else "defaults"})
        d, t = rows[-2], rows[-1]
        print(f"  {WM.DISPLAY[name]:38s} default test F1 {d['Test F1']:.4f} (gap {d['Overfit gap (train-test F1)']:+.4f})"
              f" -> tuned {t['Test F1']:.4f} (gap {t['Overfit gap (train-test F1)']:+.4f})  ({time.time() - t0:.0f}s)")
        print(f"      best: {best[name]}")
    pd.DataFrame(rows).round(4).to_csv(OUT / "table_r5_tuning.csv", index=False)
    WM.TUNED.write_text(json.dumps(best, indent=2))
    return best, rows, searches


# ------------------------------------------------------------------ 4. validation curves
def curves(X, y, tr, searches) -> dict:
    print("\n=== R5.4 validation curves ===")
    Xt, yt = X.iloc[tr], y[tr]
    out = {}
    cr = pd.DataFrame(searches["lr"].cv_results_)
    cr = cr[(cr["param_clf__l1_ratio"] == 0.0) & (cr["param_sel"].astype(str) == "passthrough")]
    out["lr_C"] = {"param": "C (inverse L2 strength)", "values": cr["param_clf__C"].astype(float).tolist(),
                   "train": cr["mean_train_score"].tolist(), "cv": cr["mean_test_score"].tolist()}
    for name in ("char_svm", "hybrid"):
        cr = pd.DataFrame(searches[name].cv_results_)
        out[f"{name}_C"] = {"param": "C", "values": cr["param_clf__C"].astype(float).tolist(),
                            "train": cr["mean_train_score"].tolist(), "cv": cr["mean_test_score"].tolist()}
    depths = [2, 3, 4, 6, 8, 10, 14]
    est = WM.build(profile=True, tuned=False, names=["xgb"])["xgb"]
    trs, cvs = validation_curve(est, Xt, yt, param_name="clf__max_depth", param_range=depths, scoring="f1",
                                cv=StratifiedKFold(3, shuffle=True, random_state=RS), n_jobs=1)
    out["xgb_depth"] = {"param": "max_depth", "values": depths, "train": trs.mean(1).tolist(), "cv": cvs.mean(1).tolist()}
    for k, v in out.items():
        print(f"  {k:12s} " + "  ".join(f"{p:g}: {a:.3f}/{b:.3f}" for p, a, b in zip(v["values"], v["train"], v["cv"])))
    return out


def payload_ablation() -> list:
    """Engineered features target payload evasion, so test them where payloads are the data:
    attack-type classification on the HttpParamsDataset training split (5-fold CV, macro-F1)."""
    print("\n=== R5.5 payload-level ablation (HttpParams train split, 5-fold CV, macro-F1) ===")
    ptrain, _ = datasets.load_httpparams()
    X = WM.to_frame([F.payload_request(v) for v in ptrain["payload"]])
    y = ptrain["attack_type"].to_numpy()
    skf = StratifiedKFold(5, shuffle=True, random_state=RS)
    rows = []
    for model, name in (("LR", "lr"), ("XGBoost", "xgb")):
        for fs, feats in (("base features (35)", F.BASE_FEATURES), ("+ engineered (44)", F.FEATURES)):
            r = cross_validate(WM.build(tuned=False, features=feats, names=[name])[name], X, y, cv=skf,
                               scoring=["f1_macro", "accuracy"], n_jobs=1 if name == "xgb" else 5)
            rows.append({"Model": model, "Feature set": fs, "CV macro-F1": float(r["test_f1_macro"].mean()),
                         "CV macro-F1 std": float(r["test_f1_macro"].std()),
                         "CV accuracy": float(r["test_accuracy"].mean())})
            print(f"  {model:8s} {fs:22s} macro-F1 {rows[-1]['CV macro-F1']:.4f} +/- {rows[-1]['CV macro-F1 std']:.4f}")
    pd.DataFrame(rows).round(4).to_csv(OUT / "table_r5_payload_ablation.csv", index=False)
    return rows


def main():
    import sys
    if "--payload-only" in sys.argv:
        p = OUT / "r5_preprocessing.json"
        res = json.loads(p.read_text()) if p.exists() else {}
        res["payload_ablation"] = payload_ablation()
        p.write_text(json.dumps(res, indent=2, default=str))
        return
    t0 = time.time()
    datasets.download()
    df = datasets.load_csic()
    X = WM.to_frame(list(df["request"]))
    y = df["label"].to_numpy()
    ptrain, ptest = datasets.load_httpparams()
    tr, te = split(y)
    res = {"eda": eda(df, X, y, tr, ptrain, ptest)}
    res["ablation"] = ablation(X, y, tr)
    res["best_params"], res["tuning"], searches = tune(X, y, tr, te)
    res["curves"] = curves(X, y, tr, searches)
    res["payload_ablation"] = payload_ablation()
    (OUT / "r5_preprocessing.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min -> results/webreq/r5_preprocessing.json, tuned_params.json")


if __name__ == "__main__":
    main()

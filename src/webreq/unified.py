"""
unified.py -- one combined HTTP-request dataset from two sources ten years apart, and the
experiments that show whether combining them is sound
(run from src/:  python -m webreq.unified)

  CSIC 2010   34,593 unique requests to a Spanish e-commerce application (generated traffic)
  SR-BH 2020  real requests to a WordPress server exposed to the Internet in July 2020
              (Sureda Riera et al., Computers & Security 2022), CAPEC multi-labels

How the two are combined (standard dataset-merging practice):
  1. common schema   source, year, method, path, query, body, content-type, label (0 normal /
                     1 attack), attack type; everything else, including tool and session
                     headers (User-Agent, Cookie, Host, ...), is dropped, so no column exists
                     that only one source fills
  2. label harmonisation
                     CSIC: normal / anomalous. SR-BH: "000 - Normal" vs any CAPEC class; the
                     primary CAPEC class maps onto the project's types
                     (66 -> sqli, 126 -> path-traversal, 88/248 -> cmdi, 242 code injection ->
                     code-injection (includes XSS), other classes -> other)
  3. de-duplication  on (method, path, query, body) within and across sources; requests that
                     carry both labels are dropped
  4. balance         SR-BH is sampled (fixed seed) to SRBH_SAMPLE unique requests, half normal
                     and half attack (attacks stratified by type), so neither source dominates
                     and the source's attack share cannot act as a shortcut
  5. exclusions      requests labelled only CAPEC-16 (dictionary password attack) are removed:
                     one guess is content-identical to a normal login
Experiments:
  U1 pooled          stratified (source x label) 80/20 split, results overall and per source
  U2 cross-source    train on one source, test on the other: a new application, 10 years later
  U3 source shortcut can a model tell the sources apart, and how far does "which source is it"
                     alone get? (the source must not predict the label)
"""
from __future__ import annotations

import json
import re
import time
from urllib.parse import unquote_plus, urlsplit

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import config
import metrics as Mx
from webreq import datasets, models as WM
from webreq.httpreq import Request
from webreq.signatures import SignatureDetector

OUT = config.WEB_RESULTS
RS = config.RANDOM_STATE
SRBH_SAMPLE = 60_000
CAPEC_TYPE = [("66", "sqli"), ("126", "path-traversal"), ("88", "cmdi"), ("248", "cmdi"),
              ("242", "code-injection"), ("153", "other"), ("272", "other"), ("274", "other"),
              ("310", "other"), ("16", "other"), ("194", "other"), ("34", "other"), ("33", "other")]


def _to_request(method, uri, body, ctype) -> Request:
    uri = str(uri or "/").strip()
    parts = uri.split(" ")
    if len(parts) >= 2 and parts[0].isalpha() and parts[0].isupper():   # "GET /x HTTP/1.1"
        method, uri = parts[0], parts[1]
    if uri.startswith(("http://", "https://")):
        u = urlsplit(uri)
        path, query = u.path or "/", u.query
    else:
        path, _, query = uri.partition("?")
    headers = {"content-type": str(ctype)} if isinstance(ctype, str) and ctype and ctype != "nan" else {}
    body = "" if not isinstance(body, str) or body == "nan" else body
    return Request(str(method or "GET").upper(), path or "/", query, body, headers)


def load_srbh(sample: int | None = SRBH_SAMPLE) -> pd.DataFrame:
    path = datasets.SRBH_DIR / datasets.SRBH_FILE
    head = pd.read_csv(path, nrows=0).columns
    labels = [c for c in head if c.split(" ")[0].isdigit()]
    use = ["request_http_method", "request_http_request", "request_body", "request_content_type"] + labels
    d = pd.read_csv(path, usecols=use, dtype=str, keep_default_na=False, low_memory=False)
    L = d[labels].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    normal_col = next(c for c in labels if c.startswith("000"))
    attack_cols = [c for c in labels if c != normal_col]
    attack = (L[attack_cols].sum(axis=1) > 0).astype(int)
    # CAPEC-16 (dictionary password attack) is a property of *many* login requests, not of one:
    # a single guess "log=rafael&pwd=975cU" is content-identical to a normal login. Requests whose
    # only label is CAPEC-16 cannot be judged per request and are excluded (not relabelled).
    c16 = next((c for c in attack_cols if c.startswith("16 ")), None)
    only16 = (L[c16] == 1) & (L[[c for c in attack_cols if c != c16]].sum(axis=1) == 0) if c16 else False
    keep = ~np.asarray(only16, bool) if c16 else np.ones(len(L), bool)
    d, L, attack = d[keep], L[keep], attack[keep]

    types = pd.Series("norm", index=L.index, dtype=object)       # first matching CAPEC class wins
    for code, t in reversed(CAPEC_TYPE):
        col = next((c for c in labels if c.split(" ")[0] == code), None)
        if col:
            types[L[col] == 1] = t
    out = pd.DataFrame({"request": [_to_request(m, u, b, c) for m, u, b, c in
                                    zip(d["request_http_method"], d["request_http_request"],
                                        d["request_body"], d["request_content_type"])],
                        "label": attack.to_numpy(), "attack_type": np.where(attack == 1, types, "norm")})
    out["source"], out["year"] = "SR-BH 2020", 2020
    out.attrs["n_raw"] = len(out)
    out = _dedupe(out)
    out.attrs["n_unique"] = len(out)
    # Label cleaning. Inspection showed that about half of the unique "normal" requests are
    # OWASP ZAP scan traffic (markers zApPX..., *.owasp.org callbacks, "Set-cookie: Tamper",
    # ";cat /etc/passwd;", backup-file probes), mostly with the payload in the URL path, which
    # the original labelling did not inspect. A "normal" request is dropped (not relabelled) if
    # it matches these explicit, detector-independent criteria.
    sus = out["label"].eq(0) & out["request"].map(suspicious_normal)
    out.attrs["n_normal_unique"] = int(out["label"].eq(0).sum())
    out.attrs["n_normal_mislabelled_removed"] = int(sus.sum())
    out = out[~sus.to_numpy()]
    out.attrs["n_capec16_only_removed"] = int(np.sum(~keep))
    if sample and len(out) > sample:
        # equal numbers of normal and attack requests (attacks stratified by type), so the
        # source's attack share (CSIC 46%) cannot be used as a shortcut after merging
        k = min(sample // 2, int((out["label"] == 0).sum()), int((out["label"] == 1).sum()))
        norm = out[out["label"] == 0].sample(k, random_state=RS)
        att = out[out["label"] == 1]
        att, _ = train_test_split(att, train_size=k, stratify=att["attack_type"], random_state=RS)
        attrs = dict(out.attrs)
        out = pd.concat([norm, att])
        out.attrs.update(attrs)
    return out.reset_index(drop=True)


_INJ = set("'\"<>;()|&`$\\{}[]*!")
_SCANNER = re.compile(r"zapp?x|0w45pz4p|owasp\.org|set-cookie:|tamper=|/etc/passwd|win\.ini|"
                      r"\.(old|bak|backup|orig|7z|zip|tar|gz|rar|swp|tmp)(\b|$)|%00|\x00|\.\./", re.I)


def suspicious_normal(r: Request) -> bool:
    """Injection characters anywhere in the decoded URL or parameters, or an OWASP ZAP marker."""
    path = unquote_plus(unquote_plus(r.path))
    text = path + "?" + unquote_plus(r.query) + " " + r.body
    return (bool(_SCANNER.search(text)) or any(c in _INJ for c in path)
            or any(any(c in _INJ for c in v) for _, _, v in r.params()))


def _key(r: Request) -> str:
    return f"{r.method} {r.path}?{r.query}#{r.body}"


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    k = df["request"].map(_key)
    nlab = df.groupby(k)["label"].transform("nunique")
    keep = (nlab == 1) & ~k.duplicated()
    return df[keep.to_numpy()]


def load() -> pd.DataFrame:
    csic = datasets.load_csic()
    csic = pd.DataFrame({"request": csic["request"], "label": csic["label"],
                         "attack_type": np.where(csic["label"] == 1, "unspecified", "norm"),
                         "source": "CSIC 2010", "year": 2010})
    srbh = load_srbh()
    both = _dedupe(pd.concat([csic, srbh], ignore_index=True)).reset_index(drop=True)
    both.attrs["srbh_raw"], both.attrs["srbh_unique"] = srbh.attrs.get("n_raw"), srbh.attrs.get("n_unique")
    both.attrs["srbh_cleaning"] = {k: srbh.attrs.get(k) for k in ("n_normal_unique", "n_normal_mislabelled_removed",
                                                                 "n_capec16_only_removed")}
    return both


def _metrics(y, pred, score) -> dict:
    m = Mx.binary_metrics(y, pred, score)
    return {k: m[k] for k in ("f1", "precision", "recall", "fpr", "roc_auc")}


def main():
    t0 = time.time()
    datasets.download()
    datasets.download_srbh()
    df = load()
    X = WM.to_frame(list(df["request"]))
    y, src = df["label"].to_numpy(), df["source"].to_numpy()
    sources = ["CSIC 2010", "SR-BH 2020"]
    res = {"composition": {s: {"requests": int((src == s).sum()), "attacks": int(y[src == s].sum()),
                               "attack_share": float(y[src == s].mean()),
                               "types": df.loc[src == s, "attack_type"].value_counts().to_dict()}
                           for s in sources},
           "srbh_raw": df.attrs.get("srbh_raw"), "srbh_unique": df.attrs.get("srbh_unique"),
           "srbh_sample": SRBH_SAMPLE, "srbh_cleaning": df.attrs.get("srbh_cleaning")}
    print("Unified dataset:", json.dumps(res["composition"], indent=1))

    # U1 pooled
    print("\n=== U1 pooled: one model trained on both sources ===")
    strat = np.char.add(src.astype(str), y.astype(str))
    tr, te = train_test_split(np.arange(len(y)), test_size=config.TEST_SIZE, stratify=strat, random_state=RS)
    sig = np.asarray(SignatureDetector().detect(list(df["request"]))[0])
    res["U1_pooled"] = {}
    rows = []
    variants = [(n, True, m) for n, m in WM.build(profile=True, names=["lr", "xgb", "char_svm", "hybrid"]).items()]
    variants += [(n, False, m) for n, m in WM.build(profile=False, names=["lr", "xgb", "hybrid"]).items()]
    for name, prof, mdl in variants:
        label = name + ("" if name == "char_svm" else (" +profile" if prof else " no-profile"))
        mdl.fit(X.iloc[tr], y[tr])
        pred, s = mdl.predict(X.iloc[te]), WM.score(mdl, X.iloc[te])
        r = {"all": _metrics(y[te], pred, s)}
        for so in sources:
            m = src[te] == so
            r[so] = _metrics(y[te][m], pred[m], s[m])
        res["U1_pooled"][label] = r
        rows.append({"Setting": "pooled (80/20)", "Model": WM.DISPLAY[name] + ("" if name == "char_svm" else
                     (" + learned profile" if prof else "")), "F1 all": r["all"]["f1"],
                     "F1 CSIC 2010": r["CSIC 2010"]["f1"], "F1 SR-BH 2020": r["SR-BH 2020"]["f1"],
                     "FPR CSIC 2010": r["CSIC 2010"]["fpr"], "FPR SR-BH 2020": r["SR-BH 2020"]["fpr"]})
        print(f"  {label:20s} all F1 {r['all']['f1']:.4f} | CSIC {r['CSIC 2010']['f1']:.4f} "
              f"(FPR {r['CSIC 2010']['fpr']:.4f}) | SR-BH {r['SR-BH 2020']['f1']:.4f} (FPR {r['SR-BH 2020']['fpr']:.4f})")
    r = {"all": _metrics(y[te], sig[te], sig[te].astype(float))}
    for so in sources:
        m = src[te] == so
        r[so] = _metrics(y[te][m], sig[te][m], sig[te][m].astype(float))
    res["U1_pooled"]["signature"] = r
    rows.append({"Setting": "pooled (80/20)", "Model": WM.DISPLAY["signature"], "F1 all": r["all"]["f1"],
                 "F1 CSIC 2010": r["CSIC 2010"]["f1"], "F1 SR-BH 2020": r["SR-BH 2020"]["f1"],
                 "FPR CSIC 2010": r["CSIC 2010"]["fpr"], "FPR SR-BH 2020": r["SR-BH 2020"]["fpr"]})
    print(f"  {WM.DISPLAY['signature']:38s} all F1 {r['all']['f1']:.4f} | CSIC {r['CSIC 2010']['f1']:.4f} | "
          f"SR-BH {r['SR-BH 2020']['f1']:.4f}")

    # U2 cross-source (profile features describe one application, so they are off here)
    print("\n=== U2 cross-source: train on one source, test on the other ===")
    res["U2_cross_source"] = {}
    for a, b in ((sources[0], sources[1]), (sources[1], sources[0])):
        ia, ib = np.where(src == a)[0], np.where(src == b)[0]
        for name, mdl in WM.build(profile=False, names=["xgb", "char_svm", "hybrid"]).items():
            mdl.fit(X.iloc[ia], y[ia])
            r = _metrics(y[ib], mdl.predict(X.iloc[ib]), WM.score(mdl, X.iloc[ib]))
            res["U2_cross_source"][f"{a} -> {b} | {name}"] = r
            rows.append({"Setting": f"train {a} -> test {b}", "Model": WM.DISPLAY[name],
                         "F1 all": r["f1"], f"F1 {b}": r["f1"], f"FPR {b}": r["fpr"]})
            print(f"  {a} -> {b}  {WM.DISPLAY[name]:38s} F1 {r['f1']:.4f}  P {r['precision']:.4f}  "
                  f"R {r['recall']:.4f}  FPR {r['fpr']:.4f}  AUC {r['roc_auc']:.4f}")
        r = _metrics(y[ib], sig[ib], sig[ib].astype(float))
        res["U2_cross_source"][f"{a} -> {b} | signature"] = r
        print(f"  {a} -> {b}  {WM.DISPLAY['signature']:38s} F1 {r['f1']:.4f}  FPR {r['fpr']:.4f}")

    # U3 source shortcut
    print("\n=== U3 source shortcut ===")
    ys = (src == sources[1]).astype(int)
    m = WM.build(profile=False, names=["xgb"])["xgb"].fit(X.iloc[tr], ys[tr])
    src_acc = float((m.predict(X.iloc[te]) == ys[te]).mean())
    base = {so: float(y[src == so].mean()) for so in sources}
    src_only = np.array([1 if base[s] >= 0.5 else 0 for s in src[te]])
    res["U3_source_shortcut"] = {
        "source_identifiable_accuracy": src_acc,
        "attack_share_by_source": base,
        "f1_using_source_only": float(f1_score(y[te], src_only, zero_division=0)),
        "roc_auc_using_source_only": float(roc_auc_score(y[te], [base[s] for s in src[te]])),
    }
    print(f"  a model can tell the source apart with accuracy {src_acc:.3f}; attack share by source {base}")
    print(f"  knowing only the source gives ROC-AUC {res['U3_source_shortcut']['roc_auc_using_source_only']:.3f} "
          f"(0.5 = no information): the source does {'not ' if res['U3_source_shortcut']['roc_auc_using_source_only'] < 0.6 else ''}predict the label")

    (OUT / "unified.json").write_text(json.dumps(res, indent=2, default=float))
    pd.DataFrame(rows).round(4).to_csv(OUT / "table_unified.csv", index=False)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min -> results/webreq/unified.json")


if __name__ == "__main__":
    main()

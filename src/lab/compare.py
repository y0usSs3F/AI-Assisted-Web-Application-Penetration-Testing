"""
compare.py -- AI prototype vs. OWASP ZAP vs. signature rules on the lab traffic.

Input: a run directory written by lab.run_lab (requests.jsonl from the logging proxy and
ZAP's zap_report.json). Ground truth per request:
  benign       phase A, simulated shoppers
  explore      phase B without X-ZAP-Scan-ID: ZAP's requestor/spider (crawling, no payloads)
  <category>   phase B with X-ZAP-Scan-ID = id of the ZAP active-scan rule that sent it,
               mapped to sqli / xss / path-traversal / cmdi / other-injection / probe
The analyzer never sees X-ZAP-Scan-ID (it is in IGNORED_HEADERS) nor the User-Agent.

Detectors compared on the same requests:
  AI prototype            webreq.analyzer (payload ensemble: char n-gram SVM + XGBoost)
  Signature rules         WAF-style rules modelled on OWASP CRS families (conventional)
  CSIC request models     XGBoost / char-SVM trained on whole CSIC 2010 requests (domain shift)
and, as the conventional vulnerability scanner, ZAP's own alerts.

    python -m lab.compare                 # latest run
    python -m lab.compare lab/runs/<ts>   # a specific run
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from webreq import models as WM
from webreq.analyzer import Analyzer
from webreq.httpreq import Request
from webreq.signatures import SignatureDetector

OUT = config.LAB_RESULTS

# ZAP active-scan rule ids (ZAP 2.17 "ascanrules"/"ascanrulesBeta") -> attack category
PLUGIN_CATEGORY = {
    **{p: "sqli" for p in (40018, 40019, 40020, 40021, 40022, 40024, 40027, 40033)},
    **{p: "xss" for p in (40012, 40014, 40016, 40017, 40026)},
    **{p: "path-traversal" for p in (6, 7)},
    **{p: "cmdi" for p in (90020, 90037)},
    **{p: "other-injection" for p in (40009, 90019, 90023, 40003, 90021, 40043, 90035, 90036,
                                      30001, 30002, 30003, 40008, 20019, 40046, 90017, 90025,
                                      40044, 40045, 40047, 10048, 20018, 90029)},
}
PLUGIN_NAME = {
    0: "Directory Browsing", 6: "Path Traversal", 7: "Remote File Inclusion",
    10045: "Source Code Disclosure - /WEB-INF", 10048: "Remote Code Execution - Shell Shock",
    10058: "GET for POST", 10095: "Backup File Disclosure", 10104: "User Agent Fuzzer",
    10106: "HTTP Only Site", 10107: "Httpoxy - Proxy Header Misuse", 20012: "Anti-CSRF Tokens Check",
    20014: "HTTP Parameter Pollution", 20015: "Heartbleed", 20016: "Cross-Domain Misconfiguration",
    20017: "Source Code Disclosure - CVE-2012-1823", 20018: "Remote Code Execution - CVE-2012-1823",
    20019: "External Redirect", 30001: "Buffer Overflow", 30002: "Format String Error",
    30003: "Integer Overflow Error", 40003: "CRLF Injection", 40008: "Parameter Tampering",
    40009: "Server Side Include", 40012: "Cross Site Scripting (Reflected)",
    40014: "Cross Site Scripting (Persistent)", 40016: "Cross Site Scripting (Persistent) - Prime",
    40017: "Cross Site Scripting (Persistent) - Spider", 40018: "SQL Injection",
    40019: "SQL Injection - MySQL", 40020: "SQL Injection - Hypersonic SQL",
    40021: "SQL Injection - Oracle", 40022: "SQL Injection - PostgreSQL",
    40024: "SQL Injection - SQLite", 40026: "Cross Site Scripting (DOM Based)",
    40027: "SQL Injection - MsSQL", 40028: "ELMAH Information Leak", 40029: "Trace.axd Information Leak",
    40032: ".htaccess Information Leak", 40033: "NoSQL Injection - MongoDB", 40034: ".env Information Leak",
    40035: "Hidden File Finder", 40042: "Spring Actuator Information Leak", 40043: "Log4Shell",
    40044: "Exponential Entity Expansion", 40045: "Spring4Shell", 40046: "Server Side Request Forgery",
    40047: "Text4shell", 90017: "XSLT Injection", 90019: "Server Side Code Injection",
    90020: "Remote OS Command Injection", 90021: "XPath Injection", 90023: "XML External Entity Attack",
    90024: "Generic Padding Oracle", 90025: "Expression Language Injection", 90026: "SOAP Action Spoofing",
    90028: "Insecure HTTP Method", 90029: "SOAP XML Injection", 90034: "Cloud Metadata Potentially Exposed",
    90035: "Server Side Template Injection", 90036: "Server Side Template Injection (Blind)",
    90037: "Remote OS Command Injection (Time Based)",
}
CORE = ["sqli", "xss", "path-traversal", "cmdi"]          # the four classes the AI is trained on
CATS = CORE + ["other-injection", "probe"]


def plugin_of(scan_id) -> int | None:
    try:
        return int(str(scan_id).strip().split(",")[0])
    except (TypeError, ValueError):
        return None


def truth(rec: dict) -> str:
    if rec["phase"] == "benign":
        return "benign"
    pid = plugin_of(rec.get("zap_scan_id"))
    if pid is None:
        return "explore"
    return PLUGIN_CATEGORY.get(pid, "probe")


PAYLOAD_CHARS = set("'\"<>;()|&`$\\{}[]*=%!")


def has_payload(req: Request) -> bool:
    """Does the request carry injection characters in any parameter value? Defined from
    characters alone, independently of every detector (cf. the payload filter of the
    CIC-IDS2017 audit). ZAP also sends payload-free requests: reflection canaries such as
    '0W45pz4p', numeric fuzzing and baseline requests."""
    for _, _, v in req.params():
        if any(c in PAYLOAD_CHARS for c in v) or ".." in v or "\x00" in v:
            return True
    return False


def load_run(run_dir: Path) -> pd.DataFrame:
    recs = [json.loads(line) for line in open(run_dir / "requests.jsonl", encoding="utf-8") if line.strip()]
    recs = [r for r in recs if r["phase"] in ("benign", "zap")]
    d = pd.DataFrame(recs)
    d["truth"] = [truth(r) for r in recs]
    d["plugin"] = [plugin_of(r.get("zap_scan_id")) for r in recs]
    d["request"] = [Request.from_dict(r) for r in recs]
    d["payload"] = [has_payload(r) for r in d["request"]]
    return d


def score(d: pd.DataFrame) -> tuple[pd.DataFrame, list, dict]:
    reqs = list(d["request"])
    timing = {}
    t0 = time.perf_counter()
    findings = Analyzer.load().analyze_many(reqs)
    timing["ai_ms_per_request"] = 1000 * (time.perf_counter() - t0) / len(reqs)
    d["ai"] = [int(f.is_attack) for f in findings]
    d["ai_prob"] = [f.attack_prob for f in findings]
    d["ai_type"] = [f.attack_type for f in findings]
    d["ai_risk"] = [f.risk_level for f in findings]
    d["ai_param"] = [f.parameter for f in findings]
    d["endpoint"] = [f.endpoint for f in findings]
    t0 = time.perf_counter()
    pred, cats, _ = SignatureDetector().detect(reqs)
    timing["sig_ms_per_request"] = 1000 * (time.perf_counter() - t0) / len(reqs)
    d["sig"], d["sig_type"] = pred, [c or "norm" for c in cats]
    X = WM.to_frame(reqs)
    for name in ("xgb", "char_svm"):
        p = config.MODELS_DIR / f"webreq_csic_{name}.joblib"
        if p.exists():
            d[f"csic_{name}"] = joblib.load(p).predict(X).astype(int)
    return d, findings, timing


DETECTORS = {"ai": "AI prototype (payload ensemble)", "sig": "Signature rules (CRS-style)",
             "csic_xgb": "CSIC-trained XGBoost (whole request)",
             "csic_char_svm": "CSIC-trained char-SVM (whole request)"}


def detection_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, name in DETECTORS.items():
        if col not in d:
            continue
        row = {"Detector": name}
        for g in ["benign", "explore"]:
            m = d["truth"] == g
            row[f"FPR {g}"] = round(float(d.loc[m, col].mean()), 4) if m.any() else np.nan
        for c in CATS:
            m = d["truth"] == c
            row[f"Recall {c}"] = round(float(d.loc[m, col].mean()), 4) if m.any() else np.nan
        for c in CORE + ["other-injection"]:
            m = (d["truth"] == c) & d["payload"]
            row[f"Recall {c} (payload-bearing)"] = round(float(d.loc[m, col].mean()), 4) if m.any() else np.nan
        # campaign level: a ZAP rule attacking one endpoint+parameter counts as detected
        # if at least one of its requests is flagged (what a tester/SOC actually needs)
        a = d[d["truth"].isin(CORE)]
        if len(a):
            camp = a.groupby(["plugin", "endpoint"])[col].max()
            row["Campaigns detected (core)"] = round(float(camp.mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def type_agreement(d: pd.DataFrame) -> dict:
    out = {}
    for col, tcol in (("ai", "ai_type"), ("sig", "sig_type")):
        m = d["truth"].isin(CORE) & (d[col] == 1)
        out[col] = {"n_flagged": int(m.sum()),
                    "type_accuracy": round(float((d.loc[m, tcol] == d.loc[m, "truth"]).mean()), 4) if m.any() else None,
                    "confusion": {f"{t}->{p}": int(n) for (t, p), n in
                                  Counter(zip(d.loc[m, "truth"], d.loc[m, tcol])).items()}}
    return out


def plugin_table(d: pd.DataFrame) -> pd.DataFrame:
    a = d[d["plugin"].notna()]
    g = a.groupby("plugin").agg(requests=("ai", "size"), ai=("ai", "mean"), sig=("sig", "mean"),
                                category=("truth", "first")).reset_index()
    g["plugin"] = g["plugin"].astype(int)
    g["rule"] = g["plugin"].map(lambda p: PLUGIN_NAME.get(p, f"rule {p}"))
    g = g.rename(columns={"ai": "AI detection rate", "sig": "Signature detection rate"})
    return g[["plugin", "rule", "category", "requests", "AI detection rate", "Signature detection rate"]] \
        .sort_values(["category", "requests"], ascending=[True, False]).round(4)


# ------------------------------------------------------------------ ZAP's own findings
def zap_alerts(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "zap_report.json"
    if not p.exists():
        return pd.DataFrame()
    rep = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for site in rep.get("site", []):
        for al in site.get("alerts", []):
            for inst in al.get("instances", []) or [{}]:
                uri = inst.get("uri", "")
                path = "/" + uri.split("/", 3)[3] if uri.count("/") >= 3 else "/"
                path, _, query = path.partition("?")
                rows.append({"plugin": int(al.get("pluginid", -1)), "alert": al.get("alert") or al.get("name"),
                             "risk": (al.get("riskdesc") or "").split(" ")[0], "cwe": al.get("cweid"),
                             "method": inst.get("method", "GET"), "uri": uri,
                             "endpoint": Request(inst.get("method", "GET"), path, query).endpoint(),
                             "param": inst.get("param", ""), "attack": inst.get("attack", "")})
    return pd.DataFrame(rows)


def zap_coverage(d: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    """For each injection vulnerability ZAP confirmed, did the AI flag the attack traffic
    (same rule, same endpoint) and give it the right class?"""
    if z.empty:
        return pd.DataFrame()
    zi = z[z["plugin"].map(lambda p: PLUGIN_CATEGORY.get(p) in CORE + ["other-injection"])]
    zi = zi.drop_duplicates(["plugin", "endpoint", "param"])
    rows = []
    for _, r in zi.iterrows():
        m = (d["plugin"] == r["plugin"]) & (d["endpoint"] == r["endpoint"])
        cat = PLUGIN_CATEGORY[r["plugin"]]
        rows.append({"ZAP alert": r["alert"], "risk": r["risk"], "CWE": r["cwe"], "endpoint": r["endpoint"],
                     "param": r["param"], "category": cat, "attack requests": int(m.sum()),
                     "AI flagged": int(d.loc[m, "ai"].sum()),
                     "AI correct class": int(((d.loc[m, "ai"] == 1) & (d.loc[m, "ai_type"] == cat)).sum()),
                     "Signatures flagged": int(d.loc[m, "sig"].sum())})
    return pd.DataFrame(rows)


def examples(d: pd.DataFrame, n=8) -> dict:
    def fmt(r):
        q = f"?{r['query']}" if r["query"] else ""
        return {"truth": r["truth"], "plugin": None if pd.isna(r["plugin"]) else int(r["plugin"]),
                "request": f"{r['method']} {r['path']}{q} {str(r['body'])[:160]}".strip()[:300],
                "ai_prob": round(float(r["ai_prob"]), 3), "ai_type": r["ai_type"],
                "parameter": r["ai_param"], "signature": int(r["sig"])}
    fp = d[d["truth"].isin(["benign", "explore"]) & (d["ai"] == 1)]
    fn = d[d["truth"].isin(CORE) & (d["ai"] == 0)]
    sig_fp = d[d["truth"].isin(["benign", "explore"]) & (d["sig"] == 1)]
    return {"ai_false_positives": [fmt(r) for _, r in fp.head(n).iterrows()],
            "ai_false_negatives": [fmt(r) for _, r in fn.sample(min(n, len(fn)), random_state=0).iterrows()],
            "signature_false_positives": [fmt(r) for _, r in sig_fp.head(n).iterrows()],
            "ai_fn_by_plugin": {PLUGIN_NAME.get(int(k), str(k)): int(v) for k, v in fn["plugin"].value_counts().items()}}


def run(run_dir: Path):
    run_dir = Path(run_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_run(run_dir)
    print(f"  {len(d):,} logged requests: {dict(Counter(d['truth']))}")
    d, findings, timing = score(d)
    det = detection_table(d)
    print(det.to_string(index=False))
    plug = plugin_table(d)
    z = zap_alerts(run_dir)
    cov = zap_coverage(d, z)

    attack_findings = [f for f, t in zip(findings, d["truth"]) if t not in ("benign", "explore")]
    prio = Analyzer.prioritise(attack_findings)
    zap_summary = {}
    if not z.empty:
        u = z.drop_duplicates(["plugin", "endpoint", "param"])
        zap_summary = {"alert_types": int(z["plugin"].nunique()), "alert_instances": len(z),
                       "unique_findings": len(u),
                       "by_risk": {k: int(v) for k, v in u["risk"].value_counts().items()},
                       "injection_findings": int(u["plugin"].map(lambda p: p in PLUGIN_CATEGORY).sum()),
                       "alerts": sorted({f"{a} [{r}]" for a, r in zip(z["alert"], z["risk"])})}
    log = (run_dir / "zap.log").read_text(errors="replace") if (run_dir / "zap.log").exists() else ""
    ts = d["ts"].to_numpy()
    zap_ts = d.loc[d["phase"] == "zap", "ts"].to_numpy()
    summary = {
        "run": run_dir.name,
        "requests": {k: int(v) for k, v in Counter(d["truth"]).items()},
        "payload_bearing": {k: int(v) for k, v in Counter(d.loc[d["payload"], "truth"]).items()},
        "zap_rules_seen": int(d["plugin"].nunique()),
        "zap_scan_minutes": round(float(zap_ts.max() - zap_ts.min()) / 60, 1) if len(zap_ts) else None,
        "timing": {k: round(v, 3) for k, v in timing.items()},
        "detection": det.to_dict(orient="records"),
        "type_agreement": type_agreement(d),
        "zap": zap_summary,
        "zap_injection_coverage": cov.to_dict(orient="records"),
        "ai_top_priorities": prio.head(15).to_dict(orient="records"),
        "examples": examples(d),
        "zap_log_tail": log.splitlines()[-15:],
        "lab_duration_minutes": round(float(ts.max() - ts.min()) / 60, 1),
    }
    (OUT / "lab_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    det.to_csv(OUT / "table_lab_detection.csv", index=False)
    plug.to_csv(OUT / "table_lab_zap_rules.csv", index=False)
    prio.to_csv(OUT / "table_lab_ai_priorities.csv", index=False)
    if not z.empty:
        z.to_csv(OUT / "zap_alerts.csv", index=False)
        cov.to_csv(OUT / "table_lab_zap_coverage.csv", index=False)
    d.drop(columns=["request", "headers"]).to_csv(OUT / "lab_scored_requests.csv.gz", index=False)
    print(f"\n  type agreement: {summary['type_agreement']['ai']['type_accuracy']}  "
          f"ZAP: {zap_summary.get('unique_findings')} findings  ->  {OUT}")
    if not cov.empty:
        print(cov.to_string(index=False))
    return summary


def latest_run() -> Path:
    runs = sorted((config.LAB_DIR / "runs").glob("*/requests.jsonl"))
    if not runs:
        sys.exit("No lab runs found: run  python -m lab.run_lab  first.")
    return runs[-1].parent


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run())

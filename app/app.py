"""
AI-Assisted Web Application Penetration Testing -- prototype UI.

    streamlit run app/app.py

Tabs
  1. Analyze a request   paste a raw HTTP request or a URL -> attack?, class, risk, why
  2. Analyze traffic     upload a proxy log / HAR / CSV / raw requests / access log ->
                         findings and a prioritised list of vulnerable endpoints
  3. Flow-level IDS      upload a CICFlowMeter CSV -> scores from the CIC-IDS2017 models
  4. Results             the project's experiments and the lab comparison with OWASP ZAP
For authorised testing and defensive use only.
"""
from __future__ import annotations

import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from webreq.analyzer import CWE, TYPE_NAMES, Analyzer  # noqa: E402
from webreq.httpreq import Request, from_url, parse_raw, split_raw_stream  # noqa: E402

SAMPLES = ROOT / "app" / "samples"
RISK_COLOR = {"High": "#c62828", "Medium": "#ef6c00", "Low": "#f9a825", "None": "#2e7d32"}
EXAMPLES = {
    "Benign search": "GET /rest/products/search?q=apple+juice HTTP/1.1\nHost: shop.local\n\n",
    "SQL injection (login)": ("POST /rest/user/login HTTP/1.1\nHost: shop.local\nContent-Type: application/json\n\n"
                              "{\"email\":\"' OR 1=1--\",\"password\":\"x\"}"),
    "SQL injection (UNION)": "GET /rest/products/search?q=qwert%27%29%29%20UNION%20SELECT%20id%2Cemail%2Cpassword%20FROM%20Users-- HTTP/1.1\nHost: shop.local\n\n",
    "Stored XSS (feedback)": ("POST /api/Feedbacks/ HTTP/1.1\nHost: shop.local\nContent-Type: application/json\n\n"
                             "{\"comment\":\"<iframe src=\\\"javascript:alert(`xss`)\\\">\",\"rating\":5}"),
    "Path traversal": "GET /ftp/..%2f..%2f..%2fetc%2fpasswd HTTP/1.1\nHost: shop.local\n\n",
    "Command injection": "GET /api/ping?host=127.0.0.1%3Bcat%20%2Fetc%2Fpasswd HTTP/1.1\nHost: shop.local\n\n",
    "Awkward but benign": ("PUT /rest/products/3/reviews HTTP/1.1\nHost: shop.local\nContent-Type: application/json\n\n"
                           "{\"message\":\"It's OK, but the mug's handle broke :( 10/10 juice <3\",\"author\":\"sara@example.com\"}"),
}

st.set_page_config(page_title="AI-Assisted Web Pentest", page_icon="🛡️", layout="wide")


@st.cache_resource
def analyzer() -> Analyzer:
    return Analyzer.load()


# ------------------------------------------------------------------ input parsing
def parse_input(text: str) -> Request:
    text = text.strip()
    if re.match(r"^https?://|^/", text) and "\n" not in text:
        return from_url("GET", text)
    if re.match(r"^[A-Z]+ \S+$", text.split("\n")[0]) and "\n" not in text:
        m, u = text.split(" ", 1)
        return from_url(m, u)
    return parse_raw(text)


_ACCESS = re.compile(r'"(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS) (\S+) HTTP/[\d.]+"')


def parse_upload(name: str, data: bytes) -> tuple[list[Request], list | None]:
    """-> (requests, ground-truth labels if the file carries them)."""
    text = data.decode("utf-8", errors="replace")
    name = name.lower()
    if name.endswith(".har"):
        reqs = []
        for e in json.loads(text)["log"]["entries"]:
            r = e["request"]
            reqs.append(from_url(r["method"], r["url"], {h["name"]: h["value"] for h in r.get("headers", [])},
                                 (r.get("postData") or {}).get("text", "")))
        return reqs, None
    if name.endswith(".jsonl") or name.endswith(".json"):
        recs = [json.loads(line) for line in text.splitlines() if line.strip()]
        truth = None
        if recs and "phase" in recs[0]:
            from lab.compare import truth as lab_truth
            recs = [r for r in recs if r.get("phase") in ("benign", "zap")]
            truth = [lab_truth(r) for r in recs]
        return [Request.from_dict(r) for r in recs], truth
    if name.endswith(".csv"):
        df = pd.read_csv(io.StringIO(text)).fillna("")
        cols = {c.lower(): c for c in df.columns}
        reqs = []
        for _, r in df.iterrows():
            method = str(r[cols["method"]]) if "method" in cols else "GET"
            body = str(r[cols["body"]]) if "body" in cols else ""
            if "url" in cols:
                reqs.append(from_url(method, str(r[cols["url"]]), body=body))
            else:
                reqs.append(Request(method, str(r[cols.get("path", "path")]),
                                    str(r[cols["query"]]) if "query" in cols else "", body))
        truth = list(df[cols["label"]].astype(str)) if "label" in cols else None
        return reqs, truth
    raw = split_raw_stream(text)
    if raw:
        return [parse_raw(c) for c in raw], None
    return [from_url(m, u) for m, u in _ACCESS.findall(text)], None      # Apache/Nginx access log


def badge(level: str) -> str:
    return (f"<span style='background:{RISK_COLOR.get(level, '#555')};color:white;padding:2px 10px;"
            f"border-radius:10px;font-weight:600'>{level}</span>")


# ------------------------------------------------------------------ tab 1
def tab_request():
    st.subheader("Analyze a single HTTP request")
    st.caption("Paste a raw request (request line, headers, blank line, body), a URL, or `METHOD /path?query`.")
    c1, c2 = st.columns([1, 3])
    ex = c1.radio("Examples", list(EXAMPLES), index=1)
    text = c2.text_area("HTTP request", EXAMPLES[ex], height=180, key=f"req_{ex}")
    if not text.strip():
        return
    req = parse_input(text)
    f = analyzer().analyze(req)
    st.divider()
    a, b, c, d = st.columns(4)
    a.metric("Verdict", "ATTACK" if f.is_attack else "Benign")
    b.metric("Attack probability", f"{f.attack_prob:.1%}")
    c.metric("Vulnerability class", TYPE_NAMES.get(f.attack_type, f.attack_type))
    d.markdown(f"**Risk**<br>{badge(f.risk_level)}<br><small>score {f.risk_score:.2f}</small>",
               unsafe_allow_html=True)
    st.markdown(f"**Endpoint** `{f.endpoint}` · **most suspicious input** `{f.parameter}`"
                + (f" · **{CWE[f.attack_type]}**" if f.attack_type in CWE else ""))
    st.code(f.value or "(no parameter values)", language=None)
    l, r = st.columns(2)
    with l:
        st.markdown("**Model opinions** (ensemble = mean)")
        st.dataframe(pd.DataFrame({"model": ["Char n-gram SVM (NLP)", "XGBoost (interpretable features)",
                                             "Ensemble"],
                                   "P(attack)": [f.nlp_prob, f.xgb_prob, f.attack_prob]}),
                     hide_index=True, width="stretch")
    with r:
        st.markdown("**Why**")
        if f.signatures:
            st.markdown("Signature rules (conventional WAF-style) that fired:")
            for s in f.signatures:
                st.markdown(f"- `{s}`")
        else:
            st.markdown("- no signature rule fired")
        for i in f.indicators:
            st.markdown(f"- {i}")
    with st.expander("Parsed request"):
        st.json(req.to_dict())


# ------------------------------------------------------------------ tab 2
def tab_traffic():
    st.subheader("Analyze recorded traffic and prioritise endpoints")
    st.caption("Formats: lab proxy log (.jsonl), browser/Burp/ZAP export (.har), CSV (method,url[,body][,label]), "
               "raw requests (CSIC 2010 style .txt) or an Apache/Nginx access log.")
    up = st.file_uploader("Traffic file", type=["jsonl", "json", "har", "csv", "txt", "log"])
    demo = sorted((config.LAB_DIR / "runs").glob("*/requests.jsonl"))
    src = None
    c1, c2 = st.columns(2)
    sample = next((p for p in (SAMPLES / "demo_traffic.jsonl", SAMPLES / "demo_traffic.csv") if p.exists()), None)
    if sample and c1.button("Load demo traffic (sample)"):
        st.session_state["traffic"] = (sample.name, sample.read_bytes())
    if demo and c2.button(f"Load latest lab run ({demo[-1].parent.name})"):
        st.session_state["traffic"] = (demo[-1].name, demo[-1].read_bytes())
    if up is not None:
        src = (up.name, up.getvalue())
    elif "traffic" in st.session_state:
        src = st.session_state["traffic"]
    if not src:
        return
    reqs, truth = parse_upload(*src)
    if not reqs:
        st.warning("No requests found in the file.")
        return
    with st.spinner(f"Scoring {len(reqs):,} requests ..."):
        findings = analyzer().analyze_many(reqs)
    df = pd.DataFrame([f.to_dict() for f in findings])
    df.insert(0, "request", [f"{r.method} {r.path}" + (f"?{r.query}" if r.query else "") for r in reqs])
    if truth is not None:
        df.insert(1, "ground truth", truth)
    n_att = int(df["is_attack"].sum())
    a, b, c, d = st.columns(4)
    a.metric("Requests", f"{len(df):,}")
    b.metric("Flagged as attacks", f"{n_att:,}", f"{n_att / len(df):.1%}", delta_color="off")
    c.metric("High-risk", int((df["risk_level"] == "High").sum()))
    d.metric("Endpoints attacked", df.loc[df["is_attack"], "endpoint"].nunique())

    if truth is not None:
        t = pd.Series(truth)
        benign = t.isin(["benign", "explore", "norm", "0", "normal"])
        st.info(f"Ground truth available: false-positive rate on benign traffic "
                f"**{df.loc[benign.values, 'is_attack'].mean():.2%}**, detection rate on attack traffic "
                f"**{df.loc[~benign.values, 'is_attack'].mean():.2%}** (includes ZAP probes that carry no payload).")

    st.markdown("#### Prioritised findings")
    prio = Analyzer.prioritise(findings)
    if prio.empty:
        st.success("No attacks detected.")
    else:
        st.dataframe(prio.drop(columns=["attack_type"]).style.format({"max_risk": "{:.2f}", "priority": "{:.2f}"}),
                     hide_index=True, width="stretch")
        l, r = st.columns(2)
        l.markdown("Attacks by class")
        l.bar_chart(df.loc[df["is_attack"], "attack_type"].map(TYPE_NAMES).value_counts())
        r.markdown("Attacks by risk level")
        r.bar_chart(df.loc[df["is_attack"], "risk_level"].value_counts())

    st.markdown("#### All requests")
    only = st.checkbox("Show only flagged requests", value=True)
    view = df[df["is_attack"]] if only else df
    cols = ["request"] + (["ground truth"] if truth is not None else []) + \
           ["attack_prob", "attack_type", "risk_level", "parameter", "value", "signatures"]
    st.dataframe(view[cols].head(2000), hide_index=True, width="stretch")
    st.download_button("Download findings (CSV)", df.to_csv(index=False), "findings.csv", "text/csv")


# ------------------------------------------------------------------ tab 3
def tab_flows():
    import joblib
    import models as M  # noqa: F401  (resolves pickled pipeline helpers)
    from flow_columns import normalize
    st.subheader("Flow-level detection (CIC-IDS2017 models)")
    st.caption("Upload a CICFlowMeter CSV. Models trained on CIC-IDS2017 Thursday HTTP flows score each flow; "
               "remember the validity audit: these models partly rely on a lab TCP fingerprint.")
    up = st.file_uploader("CICFlowMeter flows (.csv)", type=["csv"], key="flows_upload")
    if st.button("Load sample flows (CIC-IDS2017 test split)"):
        st.session_state["flows"] = (SAMPLES / "cic_flows_sample.csv").read_bytes()
    data = up.getvalue() if up is not None else st.session_state.get("flows")
    if not data:
        return
    raw = pd.read_csv(io.BytesIO(data), low_memory=False)
    feats = json.loads(config.FEATURES_JSON.read_text())
    X, meta, rep = normalize(raw, feats)
    st.caption(f"{len(X):,} flows · format {rep['style']} · features matched {rep['matched']}/{len(feats)}")
    det = st.selectbox("Detector", config.ML_MODELS, index=config.ML_MODELS.index("xgb"),
                       format_func=lambda k: config.DISPLAY[k])
    bm = joblib.load(config.MODELS_DIR / f"binary_{det}.joblib")
    mc = joblib.load(config.MODELS_DIR / f"mc_{det}.joblib")
    out = pd.DataFrame({"attack score": M.attack_score(bm, X), "attack": bm.predict(X), "class": mc.predict(X)})
    lab = next((c for c in raw.columns if c.strip().lower() == "label"), None)
    if lab:
        out.insert(0, "true label", raw[lab].astype(str).str.strip().values)
    a, b = st.columns(2)
    a.metric("Flows flagged", f"{int(out['attack'].sum()):,} / {len(out):,}")
    b.bar_chart(out.loc[out["attack"] == 1, "class"].value_counts())
    if lab:
        st.markdown("Predicted class vs. true label")
        st.dataframe(pd.crosstab(out["true label"], out["class"]), width="stretch")
    st.dataframe(out.head(1000), width="stretch")


# ------------------------------------------------------------------ tab 4
def _table(path: Path, title: str, note: str = ""):
    if path.exists():
        st.markdown(f"**{title}**" + (f" — {note}" if note else ""))
        st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")


def tab_results():
    R = config.RESULTS_DIR
    st.subheader("Evaluation results")
    s1, s2, s3 = st.tabs(["Request level (CSIC 2010, HttpParams)", "Lab: AI vs OWASP ZAP", "Flow level (CIC-IDS2017)"])
    with s1:
        _table(R / "webreq" / "table_r1_csic_detection.csv", "R1 Detection on unseen CSIC 2010 requests",
               "80/20 held-out split after de-duplication")
        _table(R / "webreq" / "table_r2_generalisation.csv", "R2 Cross-dataset generalisation",
               "CSIC-trained request models on HttpParamsDataset")
        _table(R / "webreq" / "table_r3_attack_types.csv", "R3 Attack-type classification (HttpParamsDataset test)")
    with s2:
        lab = config.LAB_RESULTS
        if not (lab / "lab_summary.json").exists():
            st.info("No lab run yet: `python -m lab.run_lab` (from src/).")
        else:
            s = json.loads((lab / "lab_summary.json").read_text())
            st.markdown(f"Run `{s['run']}` · requests {s['requests']} · ZAP rules exercised {s['zap_rules_seen']} · "
                        f"ZAP scan {s['zap_scan_minutes']} min · AI {s['timing']['ai_ms_per_request']:.2f} ms/request")
            _table(lab / "table_lab_detection.csv", "Detection on Juice Shop traffic")
            _table(lab / "table_lab_zap_coverage.csv", "ZAP-confirmed injection findings: did the AI see them?")
            _table(lab / "table_lab_ai_priorities.csv", "AI prioritised endpoints (attack traffic)")
            if s.get("zap"):
                st.markdown("**ZAP alert types**")
                st.write(s["zap"]["alerts"])
            _table(lab / "table_lab_zap_rules.csv", "Per ZAP scan rule")
    with s3:
        _table(R / "tables" / "table_binary.csv", "E1 Binary detection (held-out)")
        _table(R / "tables" / "table_multiclass.csv", "E2 Attack-type classification")
        _table(R / "tables" / "table_unseen_attacks.csv", "E3 Unseen attack types (leave-one-attack-out)")
        figs = sorted((R / "figures").glob("*.png"))
        if figs:
            pick = st.selectbox("Figure", figs, format_func=lambda p: p.stem)
            st.image(str(pick))


# ------------------------------------------------------------------ layout
st.title("AI-Assisted Web Application Penetration Testing")
st.caption("Detects and classifies SQL injection, XSS, path traversal and OS command injection in HTTP traffic, "
           "ranks the riskiest endpoints, and explains each decision. For authorised testing only.")
t1, t2, t3, t4 = st.tabs(["Analyze a request", "Analyze traffic", "Flow-level IDS", "Results"])
with t1:
    tab_request()
with t2:
    tab_traffic()
with t3:
    tab_flows()
with t4:
    tab_results()

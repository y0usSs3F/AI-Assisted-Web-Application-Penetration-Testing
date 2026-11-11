"""
analyzer.py -- the AI-assisted web vulnerability assessment prototype.

For each HTTP request it answers four questions a penetration tester or SOC analyst asks:
  1. Is this request an attack?   every parameter value (query, form, JSON, unusual path
                                  segments, content-bearing headers) is scored by an ensemble
                                  of two payload models: a calibrated character n-gram SVM
                                  (NLP) and XGBoost on interpretable features
  2. Which vulnerability class?   the ensemble's most likely class for the most suspicious
                                  parameter: SQLi, XSS, path traversal or OS command injection
  3. How urgent is it?            risk score = attack probability x class severity,
                                  mapped to Low / Medium / High
  4. Why?                         signature rules that fired, plus interpretable indicators
                                  far outside the range of benign parameter values
Across many requests it prioritises: endpoint/parameter pairs are ranked by the risk and the
volume of attacks aimed at them.

Why payload level? Experiment R2 showed that detectors trained on whole requests of one
application (CSIC 2010) flag almost every benign request sent to a different application:
they learn "normal for that shop". Parameter payloads transfer between applications.

    from webreq.analyzer import Analyzer
    from webreq.httpreq import parse_raw
    Analyzer.load().analyze(parse_raw("GET /search?q=1'+OR+'1'='1 HTTP/1.1\\n\\n"))
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import joblib
import numpy as np
import pandas as pd

import config
from webreq import features as F
from webreq import models as WM
from webreq.httpreq import IGNORED_HEADERS, Request
from webreq.signatures import _majority, inspect

BUNDLE = config.MODELS_DIR / "webreq_prototype.joblib"
SEVERITY = {"sqli": 1.0, "cmdi": 1.0, "path-traversal": 0.8, "xss": 0.7}
TYPE_NAMES = {"sqli": "SQL injection", "xss": "Cross-site scripting", "cmdi": "OS command injection",
              "path-traversal": "Path traversal", "norm": "None"}
CWE = {"sqli": "CWE-89", "xss": "CWE-79", "cmdi": "CWE-78", "path-traversal": "CWE-22"}
THRESHOLD = 0.5


def risk_level(score: float) -> str:
    return "High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low" if score >= 0.15 else "None"


@dataclass
class Finding:
    is_attack: bool
    attack_prob: float             # ensemble probability for the most suspicious parameter
    nlp_prob: float                # character n-gram SVM (calibrated)
    xgb_prob: float                # XGBoost on interpretable features
    attack_type: str               # sqli | xss | path-traversal | cmdi | norm
    type_confidence: float
    risk_score: float
    risk_level: str
    endpoint: str
    parameter: str
    value: str
    signatures: list = field(default_factory=list)
    indicators: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def candidates(req: Request) -> list[tuple[str, str]]:
    """(parameter name, decoded value) pairs that the payload models inspect."""
    out = [(f"{loc}:{name}", val) for loc, name, val in req.params() if val.strip()]
    out += [(f"header:{k}", v) for k, v in req.headers.items() if k not in IGNORED_HEADERS and v.strip()]
    return out or [("<none>", "")]


class Analyzer:
    def __init__(self, bundle: dict):
        self.nlp = bundle["typer_nlp"]
        self.xgb = bundle["typer_xgb"]
        self.classes = list(bundle["classes"])
        self.p99 = bundle["benign_p99"]

    @classmethod
    def load(cls, path=BUNDLE) -> "Analyzer":
        return cls(joblib.load(path))

    def _proba(self, values: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        Xv = WM.to_frame([F.payload_request(v) for v in values])
        pn = self.nlp.predict_proba(Xv)
        px = self.xgb.predict_proba(Xv)
        return pn, px, Xv

    def analyze(self, req: Request) -> "Finding":
        return self.analyze_many([req])[0]

    def analyze_many(self, reqs: list[Request]) -> list["Finding"]:
        if not reqs:
            return []
        cand = [(i, n, v) for i, r in enumerate(reqs) for n, v in candidates(r)]
        pn, px, Xv = self._proba([v for _, _, v in cand])
        ni = self.classes.index("norm")
        empty = np.array([not v.strip() for _, _, v in cand])
        for p in (pn, px):                    # nothing to inspect -> nothing to flag
            p[empty] = 0.0
            p[empty, ni] = 1.0
        pe = (pn + px) / 2
        best: dict[int, int] = {}
        for k, (i, _, _) in enumerate(cand):
            if i not in best or pe[k, ni] < pe[best[i], ni]:
                best[i] = k
        out = []
        for i, r in enumerate(reqs):
            k = best[i]
            _, pname, val = cand[k]
            p_att = float(1 - pe[k, ni])
            atk = [j for j in range(len(self.classes)) if j != ni]
            j = max(atk, key=lambda j: pe[k, j])
            atype = self.classes[j]
            is_attack = p_att >= THRESHOLD
            score = p_att * SEVERITY.get(atype, 0.6) if is_attack else p_att * 0.3
            sigs = inspect(r)
            out.append(Finding(
                is_attack=is_attack, attack_prob=p_att, nlp_prob=float(1 - pn[k, ni]),
                xgb_prob=float(1 - px[k, ni]), attack_type=atype if is_attack else "norm",
                type_confidence=float(pe[k, j] / max(p_att, 1e-9)) if is_attack else float(pe[k, ni]),
                risk_score=float(score), risk_level=risk_level(score) if is_attack else "None",
                endpoint=r.endpoint(), parameter=pname, value=val[:300],
                signatures=[f"{s.rid} {s.description}" for s in sigs],
                indicators=self._indicators(Xv.iloc[k])))
        return out

    def _indicators(self, row) -> list[str]:
        """Interpretable features far outside the range seen in benign parameter values."""
        out = []
        for f in F.GROUPS["content"] + ["max_value_len", "pct_encoded", "double_encoded"]:
            v, hi = float(row[f]), self.p99.get(f, math.inf)
            if v > hi and v > 0 and f not in ("entropy", "alpha_ratio", "digit_ratio", "n_spaces"):
                out.append(f"{F.LABELS[f]}: {v:g} (benign 99th percentile {hi:g})")
        return out[:6]

    # -------------------------------------------------------------- prioritisation
    @staticmethod
    def prioritise(findings: list["Finding"]) -> pd.DataFrame:
        rows = [f.to_dict() for f in findings if f.is_attack]
        cols = ["endpoint", "parameter", "vulnerability", "requests", "max_risk", "risk_level", "priority"]
        if not rows:
            return pd.DataFrame(columns=cols)
        d = pd.DataFrame(rows)
        g = d.groupby(["endpoint", "parameter", "attack_type"]).agg(
            requests=("risk_score", "size"), max_risk=("risk_score", "max")).reset_index()
        g["priority"] = g["max_risk"] * np.log1p(g["requests"])
        g["risk_level"] = g["max_risk"].map(risk_level)
        g["vulnerability"] = g["attack_type"].map(
            lambda t: TYPE_NAMES.get(t, t) + (f" ({CWE[t]})" if t in CWE else ""))
        return g.sort_values("priority", ascending=False)[cols + ["attack_type"]].reset_index(drop=True)


def build_bundle(typer_nlp, typer_xgb, benign_X: pd.DataFrame):
    p99 = {f: float(np.percentile(benign_X[f], 99)) for f in F.FEATURES}
    classes = list(typer_xgb.steps[-1][1].classes_)
    assert classes == list(typer_nlp.classes_), "typer models disagree on class order"
    joblib.dump({"typer_nlp": typer_nlp, "typer_xgb": typer_xgb, "classes": classes,
                 "benign_p99": p99}, BUNDLE)

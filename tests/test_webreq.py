"""
Unit tests for the request-level pipeline: parsing, feature extraction, signature rules,
the learned profile transformer, the lab ground-truth mapping and (if trained) the analyzer.
Run:  python -m pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from webreq import features as F  # noqa: E402
from webreq.engineering import CorrelationFilter, ProfileFeatures, context  # noqa: E402
from webreq.httpreq import Request, from_url, parse_raw, split_raw_stream  # noqa: E402
from webreq.signatures import inspect  # noqa: E402


# ------------------------------------------------------------------ parsing
def test_parse_raw_json_body():
    r = parse_raw("POST /rest/user/login HTTP/1.1\nHost: x\nContent-Type: application/json\n\n"
                  '{"email":"a@b.c","password":"p","nested":{"k":"v"}}')
    assert r.method == "POST" and r.path == "/rest/user/login"
    assert ("json", "nested.k", "v") in r.params() or any(v == "v" for _, _, v in r.params())
    assert r.headers["content-type"] == "application/json"


def test_split_raw_stream_csic_format():
    text = ("GET http://localhost:8080/a?x=1 HTTP/1.1\nHost: localhost\n\n"
            "POST http://localhost:8080/b HTTP/1.1\nContent-Length: 3\n\nx=2\n")
    assert len(split_raw_stream(text)) == 2


def test_query_and_path_params():
    r = from_url("GET", "/rest/products/1'--/reviews?q=apple+juice")
    p = r.params()
    assert ("query", "q", "apple juice") in p
    assert any(loc == "path" and v == "1'--" for loc, _, v in p)


def test_traversal_hidden_in_safe_segments():
    assert any(n == "<path>" for _, n, _ in from_url("GET", "/ftp/../../etc/passwd").params())
    assert not any(n == "<path>" for _, n, _ in from_url("GET", "/ftp/legal.md").params())


def test_endpoint_collapses_ids_and_payloads():
    assert from_url("GET", "/api/Products/42").endpoint() == "GET /api/Products/{id}"
    assert from_url("GET", "/rest/track-order/%27%20OR%201=1").endpoint() == "GET /rest/track-order/{value}"


def test_roundtrip_dict():
    r = from_url("PUT", "/a?b=c", {"X-Test": "1"}, "body")
    assert Request.from_dict(r.to_dict()) == r


# ------------------------------------------------------------------ features
def test_feature_vector_complete_and_finite():
    df = F.frame([from_url("GET", "/search?q=1%27%20UNION%20SELECT%201--"), from_url("GET", "/")])
    assert list(df.columns) == F.FEATURES
    assert np.isfinite(df.to_numpy(dtype=float)).all()


def test_double_encoding_decoded():
    f = F.extract(from_url("GET", "/s?q=%2527%2520or%25201%253D1"))
    assert f["double_encoded"] > 0 and f["n_quotes"] >= 1


def test_engineered_features_detect_evasion():
    e = F.engineered_features("1' UnIoN SeLeCt char(65),0x41424344--")
    assert e["mixed_case_kw"] >= 2 and e["n_obfuscation"] >= 2 and e["n_func_calls"] >= 1
    b = F.engineered_features("apple juice")
    assert b["mixed_case_kw"] == 0 and b["n_obfuscation"] == 0 and b["kw_density"] == 0


def test_tool_headers_ignored():
    a = F.extract(from_url("GET", "/x", {"User-Agent": "sqlmap/1.8 <script>", "X-ZAP-Scan-ID": "40018"}))
    b = F.extract(from_url("GET", "/x"))
    assert a == b


# ------------------------------------------------------------------ signatures
@pytest.mark.parametrize("url,cat", [
    ("/s?q=1'+OR+'1'='1", "sqli"), ("/s?q=<script>alert(1)</script>", "xss"),
    ("/f?file=../../etc/passwd", "path-traversal"), ("/p?h=127.0.0.1;cat+/etc/passwd", "cmdi")])
def test_signature_categories(url, cat):
    cats = {r.category for r in inspect(from_url("GET", url))}
    assert cat in cats, cats


def test_signatures_quiet_on_benign():
    assert inspect(from_url("GET", "/rest/products/search?q=apple")) == []


# ------------------------------------------------------------------ learned preprocessing
def _ctx(reqs):
    return pd.DataFrame({"ctx": [context(r) for r in reqs]})


def test_profile_learns_only_from_benign():
    train = [from_url("GET", f"/shop?id={i}&name=anna") for i in range(20)]
    attack = from_url("GET", "/shop?id=1'--&name=anna")
    X = _ctx(train + [attack])
    y = np.array([0] * 20 + [1])
    p = ProfileFeatures().fit(X, y)
    out = p.transform(_ctx([from_url("GET", "/shop?id=7&name=anna"), attack,
                            from_url("GET", "/admin?debug=1")]))
    normal, att, unseen = out
    assert normal[[0, 1, 3, 4]].sum() == 0 and normal[2] < 1  # fits the profile
    assert att[3] >= 2                                       # new characters ' and -
    assert unseen[0] == 1 and unseen[1] >= 1                  # new endpoint and parameter


def test_profile_accepts_string_labels():
    X = _ctx([from_url("GET", "/a?x=1")] * 6)
    p = ProfileFeatures().fit(X, np.array(["norm"] * 6))
    assert len(p.endpoints_) == 1


def test_correlation_filter_drops_duplicates_and_constants():
    a = np.arange(50, dtype=float)
    X = np.c_[a, 2 * a + 1, np.ones(50), np.sin(a)]
    cf = CorrelationFilter(0.95).fit(X)
    assert list(cf.keep_) == [0, 3]


# ------------------------------------------------------------------ lab ground truth
def test_lab_truth_mapping():
    from lab.compare import truth
    assert truth({"phase": "benign"}) == "benign"
    assert truth({"phase": "zap", "zap_scan_id": None}) == "explore"
    assert truth({"phase": "zap", "zap_scan_id": "40018"}) == "sqli"
    assert truth({"phase": "zap", "zap_scan_id": "40012"}) == "xss"
    assert truth({"phase": "zap", "zap_scan_id": "6"}) == "path-traversal"
    assert truth({"phase": "zap", "zap_scan_id": "10104"}) == "probe"


# ------------------------------------------------------------------ analyzer (needs trained bundle)
BUNDLE = ROOT / "models" / "webreq_prototype.joblib"


@pytest.mark.skipif(not BUNDLE.exists(), reason="prototype not trained (python -m webreq.experiments)")
def test_analyzer_end_to_end():
    from webreq.analyzer import Analyzer
    a = Analyzer.load()
    benign, sqli, trav = a.analyze_many([
        from_url("GET", "/rest/products/search?q=apple"),
        from_url("GET", "/rest/products/search?q=qwert%27%29%29%20UNION%20SELECT%20id,email%20FROM%20Users--"),
        from_url("GET", "/ftp/..%2f..%2fetc%2fpasswd")])
    assert not benign.is_attack and benign.risk_level == "None"
    assert sqli.is_attack and sqli.attack_type == "sqli" and sqli.risk_level == "High"
    assert trav.is_attack and trav.attack_type == "path-traversal"
    prio = Analyzer.prioritise([benign, sqli, trav])
    assert len(prio) == 2 and prio.iloc[0]["attack_type"] == "sqli"

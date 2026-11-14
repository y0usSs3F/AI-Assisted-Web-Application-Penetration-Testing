"""
End-to-end tests of the Streamlit app (app/app.py): every feature is driven the way a user
drives it -- choosing examples, typing requests, clicking buttons, uploading files, switching
detectors -- with Streamlit's AppTest harness. No browser or server is needed.

Run:  python -m pytest tests/test_app.py -v
The manual test plan for the same features is in tests/APP_TEST_PLAN.md (IDs A1..D3 match).
Needs the trained models (models/ is not committed): run src/run_all.py first.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "app.py"
FIX = ROOT / "tests" / "fixtures" / "app"
sys.path.insert(0, str(ROOT / "src"))

st_testing = pytest.importorskip("streamlit.testing.v1")
import config  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (config.MODELS_DIR / "webreq_prototype.joblib").exists()
    or not (config.MODELS_DIR / "binary_xgb.joblib").exists(),
    reason="models not trained (run src/run_all.py)")

TIMEOUT = 300
REQUEST, TRAFFIC, FLOWS, RESULTS = range(4)          # top-level tab indices


def fresh():
    """A freshly started app session, after its first run."""
    at = st_testing.AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def metrics(block) -> dict:
    return {m.label: m.value for m in block.metric}


def no_errors(at):
    assert not at.exception, [e.value for e in at.exception]


@pytest.fixture(scope="module")
def app():
    return fresh()


# ====================================================================== A. app shell
def test_A1_app_starts_with_title_and_four_tabs(app):
    assert app.title[0].value == "AI-Assisted Web Application Penetration Testing"
    assert [t.label for t in app.tabs[:4]] == ["Analyze a request", "Analyze traffic",
                                                "Flow-level IDS", "Results"]


# ====================================================================== B. analyze a request
@pytest.mark.parametrize("example,verdict,klass", [
    ("Benign search", "Benign", "None"),
    ("SQL injection (login)", "ATTACK", "SQL injection"),
    ("SQL injection (UNION)", "ATTACK", "SQL injection"),
    ("Stored XSS (feedback)", "ATTACK", "Cross-site scripting"),
    ("Path traversal", "ATTACK", "Path traversal"),
    ("Command injection", "ATTACK", "OS command injection"),
    pytest.param("Awkward but benign", "Benign", "None", marks=pytest.mark.xfail(
        strict=True, reason="known false positive: the review text with '<3' scores 52.9% XSS")),
])
def test_B1_each_example_gets_the_right_verdict_and_class(app, example, verdict, klass):
    app.tabs[REQUEST].radio[0].set_value(example).run()
    no_errors(app)
    m = metrics(app.tabs[REQUEST])
    assert m["Verdict"] == verdict
    assert m["Vulnerability class"] == klass


@pytest.mark.parametrize("text,verdict,klass", [
    # full URL
    ("http://shop.local/rest/products/search?q=' OR 1=1--", "ATTACK", "SQL injection"),
    # METHOD /path?query shorthand
    ("GET /api/ping?host=127.0.0.1;cat /etc/passwd", "ATTACK", "OS command injection"),
    # bare path
    ("/rest/products/search?q=apple", "Benign", "None"),
    # raw request with a JSON body
    ('POST /api/Feedbacks/ HTTP/1.1\nHost: x\nContent-Type: application/json\n\n'
     '{"comment":"<script>alert(document.cookie)</script>","rating":1}', "ATTACK", "Cross-site scripting"),
])
def test_B2_every_input_format_is_parsed_and_scored(app, text, verdict, klass):
    app.tabs[REQUEST].text_area[0].set_value(text).run()
    no_errors(app)
    m = metrics(app.tabs[REQUEST])
    assert (m["Verdict"], m["Vulnerability class"]) == (verdict, klass)


def test_B3_empty_input_shows_no_result(app):
    app.tabs[REQUEST].text_area[0].set_value("   ").run()
    no_errors(app)
    assert metrics(app.tabs[REQUEST]) == {}


def test_B4_attack_result_is_explained():
    at = fresh()                                        # default example: SQL injection (login)
    t = at.tabs[REQUEST]
    md = " ".join(m.value for m in t.markdown)
    assert "High" in md                                  # risk badge
    assert "/rest/user/login" in md and "json:email" in md and "CWE-89" in md
    assert t.code[0].value == "' OR 1=1--"               # the most suspicious value
    assert "SQLI-" in md                                 # a signature rule that fired
    assert "benign 99th percentile" in md                # an out-of-range feature
    opinions = t.dataframe[0].value                      # model opinions table
    assert list(opinions["model"]) == ["Char n-gram SVM (NLP)", "XGBoost (interpretable features)", "Ensemble"]
    assert opinions["P(attack)"].between(0, 1).all()
    assert abs(opinions["P(attack)"].iloc[2] - opinions["P(attack)"].iloc[:2].mean()) < 1e-6
    assert t.expander[0].label == "Parsed request"
    assert t.json[0].value and json.loads(t.json[0].value)["method"] == "POST"


def test_B5_benign_result_has_no_risk_and_no_cwe(app):
    app.tabs[REQUEST].radio[0].set_value("Benign search").run()
    md = " ".join(m.value for m in app.tabs[REQUEST].markdown)
    assert ">None<" in md and "CWE-" not in md
    assert "no signature rule fired" in md


# ====================================================================== C. analyze traffic
def upload(at, path: Path, tab=TRAFFIC):
    at.tabs[tab].file_uploader[0].upload(path.name, path.read_bytes()).run()
    no_errors(at)
    return at.tabs[tab]


@pytest.mark.parametrize("fname,n_req,min_flagged,max_flagged", [
    ("sample.har", 3, 2, 2),            # browser / Burp / ZAP export
    ("sample_urls.csv", 4, 2, 2),       # CSV with method,url,body,label
    ("sample_paths.csv", 3, 1, 1),      # CSV with method,path,query,body
    ("csic_style.txt", 2, 1, 1),        # raw requests, CSIC 2010 format
    ("access.log", 3, 1, 1),            # Apache / Nginx access log
    ("lab_mini.jsonl", 3, 1, 1),        # lab proxy log ("setup" phase is dropped)
])
def test_C1_each_upload_format_is_parsed_and_scored(fname, n_req, min_flagged, max_flagged):
    t = upload(fresh(), FIX / fname)
    m = metrics(t)
    assert m["Requests"] == str(n_req)
    assert min_flagged <= int(m["Flagged as attacks"]) <= max_flagged
    assert len(t.dataframe) >= 2                         # prioritised findings + all requests


@pytest.mark.parametrize("fname", ["sample_urls.csv", "lab_mini.jsonl"])
def test_C2_ground_truth_files_report_fpr_and_detection_rate(fname):
    t = upload(fresh(), FIX / fname)
    info = " ".join(i.value for i in t.info)
    assert "false-positive rate on benign traffic **0.00%**" in info
    assert "detection rate on attack traffic **100.00%**" in info


def test_C3_file_without_requests_warns():
    t = upload(fresh(), FIX / "empty.log")
    assert [w.value for w in t.warning] == ["No requests found in the file."]
    assert metrics(t) == {}


@pytest.mark.xfail(strict=True, reason="known issue: a malformed CSV crashes with a pandas traceback "
                                        "instead of a friendly message")
def test_C4_malformed_csv_shows_a_friendly_error():
    at = fresh()
    at.tabs[TRAFFIC].file_uploader[0].upload("malformed.csv", (FIX / "malformed.csv").read_bytes()).run()
    no_errors(at)


@pytest.fixture(scope="module")
def demo():
    at = fresh()
    btn = next(b for b in at.tabs[TRAFFIC].button if b.label.startswith("Load demo traffic"))
    btn.click().run()
    no_errors(at)
    return at


def test_C5_demo_traffic_button(demo):
    t = demo.tabs[TRAFFIC]
    m = metrics(t)
    assert m["Requests"] == "1,991"
    assert int(m["Flagged as attacks"].replace(",", "")) > 0
    assert int(m["Endpoints attacked"]) > 0
    assert any("Ground truth available" in i.value for i in t.info)
    prio = t.dataframe[0].value
    assert len(prio) > 0 and "endpoint" in " ".join(map(str, prio.columns)).lower()


def test_C6_only_flagged_checkbox_filters_the_request_table(demo):
    t = demo.tabs[TRAFFIC]
    flagged = int(metrics(t)["Flagged as attacks"].replace(",", ""))
    assert t.checkbox[0].label == "Show only flagged requests" and t.checkbox[0].value is True
    assert len(t.dataframe[-1].value) == min(flagged, 2000)
    t.checkbox[0].uncheck().run()
    no_errors(demo)
    assert len(demo.tabs[TRAFFIC].dataframe[-1].value) == min(1991, 2000)
    demo.tabs[TRAFFIC].checkbox[0].check().run()


def test_C7_findings_can_be_downloaded_as_csv(demo):
    dl = demo.tabs[TRAFFIC].get("download_button")
    assert dl, "download button missing"
    assert "Download findings (CSV)" in str(dl[0].proto.label)


@pytest.mark.skipif(not list((config.LAB_DIR / "runs").glob("*/requests.jsonl")),
                    reason="no local lab run (python -m lab.run_lab)")
def test_C8_latest_lab_run_button():
    at = fresh()
    btn = next(b for b in at.tabs[TRAFFIC].button if b.label.startswith("Load latest lab run"))
    btn.click().run()
    no_errors(at)
    assert int(metrics(at.tabs[TRAFFIC])["Requests"].replace(",", "")) > 1000


# ====================================================================== D. flow-level IDS
@pytest.fixture(scope="module")
def flows():
    at = fresh()
    at.tabs[FLOWS].button[0].click().run()               # Load sample flows
    no_errors(at)
    return at


def test_D1_sample_flows_are_scored_with_all_features_matched(flows):
    t = flows.tabs[FLOWS]
    cap = " ".join(c.value for c in t.caption)
    n_feats = len(json.loads(config.FEATURES_JSON.read_text()))
    assert f"features matched {n_feats}/{n_feats}" in cap
    flagged, total = metrics(t)["Flows flagged"].replace(",", "").split(" / ")
    assert 0 < int(flagged) <= int(total)
    crosstab = t.dataframe[0].value                       # predicted class vs. true label
    assert "BENIGN" in [str(i).upper() for i in crosstab.index]


@pytest.mark.parametrize("detector", config.ML_MODELS)
def test_D2_every_flow_detector_can_be_selected(flows, detector):
    sb = next(s for s in flows.tabs[FLOWS].selectbox if s.label == "Detector")
    sb.set_value(detector).run()
    no_errors(flows)
    assert "Flows flagged" in metrics(flows.tabs[FLOWS])


def test_D3_flow_csv_upload():
    t = upload(fresh(), config.ROOT / "app" / "samples" / "cic_flows_sample.csv", tab=FLOWS)
    assert "Flows flagged" in metrics(t)


# ====================================================================== E. results
def test_E1_results_tables_are_shown(app):
    md = " ".join(m.value for t in app.tabs[RESULTS:] for m in t.markdown)
    for title in ["R1 Detection", "R2 Cross-dataset", "R3 Attack-type",
                  "Detection on Juice Shop traffic", "AI prioritised endpoints",
                  "E1 Binary detection", "E2 Attack-type", "E3 Unseen attack types"]:
        assert title in md, title
    assert "ms/request" in md                            # lab summary line


def test_E2_every_figure_can_be_opened(app):
    sb = next(s for s in app.tabs[RESULTS].selectbox if s.label == "Figure")
    figs = sorted((config.RESULTS_DIR / "figures").glob("*.png"))
    assert len(sb.options) == len(figs) > 0
    for f in figs:
        sb.set_value(f).run()
        no_errors(app)
        assert len(app.tabs[RESULTS].image) == 1

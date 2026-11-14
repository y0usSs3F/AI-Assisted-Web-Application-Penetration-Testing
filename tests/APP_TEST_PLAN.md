# App test plan

How to test every feature of the Streamlit app ([`app/app.py`](../app/app.py)), by hand and automatically.
Each test ID (A1, B1, …) is the same in this plan and in [`test_app.py`](test_app.py).

## Before you start

The trained models must exist in `models/`, which is not in Git. If they are missing, run `cd src && python run_all.py --skip-download` first.

```bash
# automated: drives every feature, no browser needed (about 25 s)
python -m pytest tests/test_app.py -v

# manual: start the app, then open http://localhost:8501
streamlit run app/app.py
```

The sample upload files are in [`tests/fixtures/app/`](fixtures/app/).

| File | Format | Contents |
|---|---|---|
| `sample.har` | browser / Burp / ZAP export | 1 benign search, 1 SQLi login, 1 path traversal |
| `sample_urls.csv` | CSV `method,url,body,label` | 2 normal, 1 SQLi, 1 XSS, with labels |
| `sample_paths.csv` | CSV `method,path,query,body` | 2 benign, 1 command injection |
| `csic_style.txt` | raw requests (CSIC 2010 format) | 1 normal, 1 XSS |
| `access.log` | Apache / Nginx access log | 2 benign, 1 SQLi |
| `lab_mini.jsonl` | lab proxy log | 2 benign shoppers, 1 ZAP path traversal, 1 "setup" request that must be ignored |
| `empty.log` | text with no requests | nothing to analyse |
| `malformed.csv` | broken CSV (unquoted comma in a URL) | robustness check |

---

## A. App shell

| ID | Feature | Manual steps | Expected result |
|---|---|---|---|
| A1 | App starts | Run `streamlit run app/app.py` and open the page | The title "AI-Assisted Web Application Penetration Testing" and four tabs appear, with no red error box |

## B. Tab "Analyze a request"

| ID | Feature | Manual steps | Expected result |
|---|---|---|---|
| B1 | Built-in examples | Click each example on the left | Benign search → **Benign**. SQLi (login), SQLi (UNION) → **ATTACK, SQL injection**. Stored XSS → **XSS**. Path traversal → **Path traversal**. Command injection → **OS command injection**. Awkward but benign → should be **Benign**, but is a **known false positive** (ATTACK, XSS 52.9%) |
| B2 | Input formats | Paste each into the text box: (1) a full URL `http://shop.local/rest/products/search?q=' OR 1=1--`; (2) `GET /api/ping?host=127.0.0.1;cat /etc/passwd`; (3) a bare path `/rest/products/search?q=apple`; (4) a raw request with a JSON body containing `<script>` | (1) ATTACK, SQLi. (2) ATTACK, command injection. (3) Benign. (4) ATTACK, XSS |
| B3 | Empty input | Clear the text box | No verdict is shown and no error appears |
| B4 | Explanation | Select "SQL injection (login)" | Risk badge **High**. Endpoint `POST /rest/user/login`, input `json:email`, **CWE-89**. Suspicious value `' OR 1=1--`. Model-opinions table with 3 rows, where Ensemble = mean of the two models. "Why" lists `SQLI-…` rules and features above the benign 99th percentile. "Parsed request" expands to JSON with `"method": "POST"` |
| B5 | Benign result | Select "Benign search" | Risk badge **None**, no CWE, "no signature rule fired" |

## C. Tab "Analyze traffic"

| ID | Feature | Manual steps | Expected result |
|---|---|---|---|
| C1 | Upload formats | Upload each sample file in turn | Requests / flagged: `sample.har` 3 / 2. `sample_urls.csv` 4 / 2. `sample_paths.csv` 3 / 1. `csic_style.txt` 2 / 1. `access.log` 3 / 1. `lab_mini.jsonl` 3 / 1 (the setup request is dropped). Each shows a "Prioritised findings" table and an "All requests" table |
| C2 | Ground-truth scoring | Upload `sample_urls.csv` or `lab_mini.jsonl` | A blue box: false-positive rate **0.00%**, detection rate **100.00%** |
| C3 | No requests in file | Upload `empty.log` | Yellow warning "No requests found in the file." |
| C4 | Malformed file | Upload `malformed.csv` | Should show a friendly error. **Known issue:** it shows a raw pandas traceback instead |
| C5 | Demo traffic | Click "Load demo traffic (sample)" | 1,991 requests, attacks flagged, endpoints attacked > 0, ground-truth box, prioritised findings table, two bar charts (by class and by risk) |
| C6 | "Show only flagged" filter | After C5, untick "Show only flagged requests" | The bottom table grows from the flagged requests to all 1,991 |
| C7 | Download | After C5, click "Download findings (CSV)" | A `findings.csv` file downloads with one row per request |
| C8 | Latest lab run | Click "Load latest lab run (…)" (only shown if `lab/runs/` exists) | About 10,000 lab requests are scored with no error |

## D. Tab "Flow-level IDS"

| ID | Feature | Manual steps | Expected result |
|---|---|---|---|
| D1 | Sample flows | Click "Load sample flows (CIC-IDS2017 test split)" | The caption says "features matched N/N" (all features found). A "Flows flagged X / Y" metric, a bar chart, and a table of predicted class vs. true label that includes BENIGN |
| D2 | Detector choice | Pick each detector in the "Detector" box (LR, linear SVM, RF, HistGB, XGBoost, MLP) | Every detector produces results with no error |
| D3 | Flow CSV upload | Upload `app/samples/cic_flows_sample.csv` | Same kind of output as D1 |

## E. Tab "Results"

| ID | Feature | Manual steps | Expected result |
|---|---|---|---|
| E1 | Result tables | Open each sub-tab | Request level: tables R1, R2, R3. Lab: summary line with "ms/request", detection, ZAP coverage, AI priorities, ZAP alerts and per-rule tables. Flow level: E1, E2, E3 tables |
| E2 | Figures | In "Flow level", pick each entry of the "Figure" box | Each figure (fig01 … fig12) is displayed |

---

## Known issues found by these tests

The tests mark these as expected failures (`xfail`, strict). When one is fixed, that test reports "XPASS", and its marker should then be removed.

1. **B1, "Awkward but benign" is flagged as XSS (52.9%).** The `<3` in a normal product review pushes it just over the 50% threshold.
2. **C4, a malformed CSV crashes the tab** with a pandas `ParserError` traceback. `parse_upload` should catch the error and show `st.error(...)`.

A missed attack found while exploring, not yet a test: the form-encoded login `user=admin'--&pw=x` (`application/x-www-form-urlencoded`) scores **Benign at 45.3%**, a false negative.

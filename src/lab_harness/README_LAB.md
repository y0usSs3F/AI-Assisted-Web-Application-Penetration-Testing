# Practical lab: AI-assisted detection vs. OWASP ZAP

This folder runs the project's **practical component**: test the AI-assisted prototype on an
intentionally vulnerable web application and compare it with a conventional automated
scanner (OWASP ZAP).

> **Authorised, isolated lab only.** These scripts start OWASP Juice Shop and attack it with
> ZAP. Run them on a VM/network you control. Never point them at systems you don't own.

## How the two sides differ

| | OWASP ZAP (conventional) | AI prototype (this project) |
|---|---|---|
| Mode | **Active**: sends attack payloads to the app | **Passive**: watches network traffic |
| Input | The running application | Network flows (CICFlowMeter statistics) |
| Output | Confirmed vulnerabilities (by risk/CWE) | Alerts on attack-like flows |
| Answers | *What is wrong with the app?* | *Would we notice someone attacking it?* |

So the comparison reports ZAP's findings side by side with the AI's detection rate on
ZAP's own attack traffic and its false-positive rate on normal browsing.

## Requirements

- A Linux VM (Ubuntu/Kali) with `docker`, `tcpdump`, `curl`, Python 3.10+.
- The project installed and trained: `pip install -r requirements.txt`, then from `src/`:
  `python download_data.py` and `python train_eval.py` (creates `models/`).
- A flow meter: Java **CICFlowMeter** (recommended, the tool family used to build CIC-IDS2017)
  or the Python port `pip install cicflowmeter`.

## Steps

**1. Capture benign browsing and a ZAP scan** (≈15–25 min):
```bash
sudo bash capture_and_scan.sh lab_run 5
```
This produces `lab_run/benign.pcap`, `lab_run/attack.pcap`, `lab_run/zap_report.json` and `zap_report.html`.

**2. Convert the captures to flows.**
- Java CICFlowMeter (precompiled release `CICFlowMeter-4.0/bin/cfm`):
  `cfm lab_run/benign.pcap lab_run/flows_benign/` and the same for `attack.pcap`.
  Rename the CSVs to `lab_run/benign_flows.csv` and `lab_run/attack_flows.csv`.
- Python port:
  `cicflowmeter -f lab_run/benign.pcap -c lab_run/benign_flows.csv`
  `cicflowmeter -f lab_run/attack.pcap -c lab_run/attack_flows.csv`

`flow_columns.py` detects which tool produced the CSV, maps its column names onto the
CIC-IDS2017 names, and converts the Python port's **seconds** into the **microseconds**
the models were trained on. It reports any feature it had to derive or fill in.

**3. Score the flows with every detector:**
```bash
python score_flows.py lab_run/benign_flows.csv --port 3000 --phase benign --out lab_run/benign_scored.csv
python score_flows.py lab_run/attack_flows.csv --port 3000 --phase attack --out lab_run/attack_scored.csv
```

**4. Compare with ZAP:**
```bash
python compare_with_zap.py --zap-json lab_run/zap_report.json \
    --benign lab_run/benign_scored.summary.json --attack lab_run/attack_scored.summary.json \
    --out lab_run/comparison.md
```
`comparison.md` (+ `.csv`) is ready to paste into the report.
`parse_zap_report.py zap_report.json findings.csv` exports every ZAP alert, categorised by CWE.

## Caveats

- **Domain shift.** The models learned CIC-IDS2017 traffic (2017 network, Selenium attacks on
  DVWA/Apache). Juice Shop (Node.js), Docker networking and a different flow-meter version
  change the statistics, so expect more false positives than on the dataset. This is the
  same effect as experiment E5 in the paper.
- **Loopback capture.** Traffic captured on `lo` has a 65,536-byte MTU, so packet-size
  statistics differ from a real network (MTU 1,500). For realistic packet sizes, run Juice
  Shop on a second VM on a host-only network and capture on that interface (`IFACE=eth1`).
- **Phase B includes ZAP's spider.** Every flow in `attack.pcap` is counted as attack traffic,
  but the crawl at the start is benign-looking, so the detection rate is a conservative lower bound.
- **TCP fingerprint shortcut.** The paper's validity audit (E7) shows the CIC-IDS2017 models
  partly key on the attacker's TCP initial window (Linux = 29,200). ZAP in Docker on Linux
  shares it, which can inflate detection for the wrong reason. Report this when discussing results.

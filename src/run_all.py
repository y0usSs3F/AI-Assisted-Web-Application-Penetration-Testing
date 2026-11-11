"""
run_all.py -- run the complete pipeline in order (about 1 hour on a laptop).

  E1-E7  flow level, CIC-IDS2017
  R1-R5  request level, HTTP CSIC 2010 + HttpParamsDataset (R5 = preprocessing/tuning, runs
         first because R1-R4 use the hyperparameters it selects)
The practical lab (Juice Shop + OWASP ZAP) is separate: python -m lab.setup_tools; python -m lab.run_lab

    python run_all.py              # everything
    python run_all.py --skip-download
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ("download, verify, preprocess", "download_data.py"),
    ("E1/E2 detection + classification", "train_eval.py"),
    ("E3 unseen attack types", "unseen_attacks.py"),
    ("E4 timing evasion", "evasion_timing.py"),
    ("E5 unseen benign day", "cross_day.py"),
    ("E6 evaluation scope", "scope_shortcut.py"),
    ("E7 validity audit", "validity_audit.py"),
    ("R5 request-level preprocessing, feature engineering, regularisation", "-m webreq.tuning"),
    ("R1-R4 request-level detection, generalisation, attack types + prototype", "-m webreq.experiments"),
    ("figures", "make_figures.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    for label, script in STEPS:
        if args.skip_download and script == "download_data.py":
            continue
        print(f"\n{'=' * 70}\n>>> {label}  ({script})\n{'=' * 70}", flush=True)
        t = time.time()
        cmd = [sys.executable, *script.split()] if script.startswith("-m ") else [sys.executable, str(HERE / script)]
        if subprocess.run(cmd, cwd=HERE).returncode != 0:
            sys.exit(f"Step failed: {script}")
        print(f"<<< {label} done in {time.time() - t:.0f}s", flush=True)
    print(f"\nAll steps finished in {(time.time() - t0) / 60:.1f} min. Results are in ../results/")


if __name__ == "__main__":
    main()

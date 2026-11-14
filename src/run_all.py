"""
run_all.py -- run the complete pipeline in order (about 15-20 minutes on a laptop).

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
        if subprocess.run([sys.executable, str(HERE / script)], cwd=HERE).returncode != 0:
            sys.exit(f"Step failed: {script}")
        print(f"<<< {label} done in {time.time() - t:.0f}s", flush=True)
    print(f"\nAll steps finished in {(time.time() - t0) / 60:.1f} min. Results are in ../results/")


if __name__ == "__main__":
    main()

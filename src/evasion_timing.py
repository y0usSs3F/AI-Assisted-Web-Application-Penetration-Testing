"""
evasion_timing.py  --  E4: "low-and-slow" timing evasion.

Flow-based detectors lean heavily on timing statistics (durations, inter-arrival times,
packet rates). A simple, well-known evasion is to slow the attack down: insert delays
between the packets/requests of a connection so the flow looks "human". We simulate this
in feature space on the held-out ATTACK flows:
    time features   (Flow Duration, all IAT, Active, Idle statistics)  x k
    rate features   (Flow Bytes/s, Flow Packets/s, Fwd/Bwd Packets/s)  / k
    size and count features (bytes, packets, header lengths, flags)     unchanged
for slow-down factors k = 1 (no evasion) ... 100, and measure each detector's detection
rate (recall) WITHOUT retraining. Benign flows are not modified, so false alarms are
unchanged.

Caveat: this is a feature-space simulation of within-connection delays. Delays inserted
*between* separate connections would not change per-flow statistics at all (only the
flow arrival rate, which per-flow features do not capture).

Run after train_eval.py:   python evasion_timing.py
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

import config
import data as D

FACTORS = [1, 2, 5, 10, 20, 50, 100]
TIME_FEATURES = [
    "Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]
RATE_FEATURES = ["Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s"]


def slow_down(X: pd.DataFrame, k: float) -> pd.DataFrame:
    Xp = X.copy()
    t = [c for c in TIME_FEATURES if c in Xp.columns]
    r = [c for c in RATE_FEATURES if c in Xp.columns]
    Xp[t] = Xp[t] * k
    Xp[r] = Xp[r] / k
    return Xp


def main():
    df = D.load_http()
    X, labels, y = D.xy(df)
    _, te = D.split_indices(labels)
    att = te[y[te] == 1]
    Xa, la = X.iloc[att], labels[att]
    print(f"Timing evasion on {len(att)} held-out attack flows; "
          f"{sum(c in X.columns for c in TIME_FEATURES)} time + "
          f"{sum(c in X.columns for c in RATE_FEATURES)} rate features perturbed")

    names = config.ML_MODELS + [config.BASELINE]
    dets = {n: joblib.load(config.MODELS_DIR / f"binary_{n}.joblib") for n in names}
    res = {n: {"overall": {}, "by_attack": {a: {} for a in config.ATTACKS}} for n in names}
    rows = []
    for k in FACTORS:
        Xk = slow_down(Xa, k)
        for n, det in dets.items():
            pred = det.predict(Xk)
            res[n]["overall"][str(k)] = float(pred.mean())
            for a in config.ATTACKS:
                res[n]["by_attack"][a][str(k)] = float(pred[la == a].mean())
            rows.append({"Slow-down factor": k, "Detector": config.DISPLAY[n],
                         "Recall (all attacks)": pred.mean(),
                         **{f"Recall {a}": pred[la == a].mean() for a in config.ATTACKS}})
        print(f"  k={k:>3}: " + "  ".join(f"{config.SHORT[n]}={res[n]['overall'][str(k)]:.3f}" for n in names))

    (config.RESULTS_DIR / "evasion_timing.json").write_text(
        json.dumps({"factors": FACTORS, "n_attacks": int(len(att)), "recall": res}, indent=2))
    pd.DataFrame(rows).round(4).to_csv(config.TAB_DIR / "table_evasion_timing.csv", index=False)
    print("Saved results/evasion_timing.json")


if __name__ == "__main__":
    main()

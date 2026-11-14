"""
cross_day.py  --  E5: false alarms on a completely unseen day.

Detectors are trained on Thursday morning. In deployment they must run on other days,
when user behaviour, services and traffic mix differ. Monday of CIC-IDS2017 contains NO
attacks, so every alarm raised on Monday's HTTP flows is a false positive. This measures:
  * the false-positive rate on Monday vs. on the Thursday test split (temporal shift),
  * how many false alerts that means in absolute terms, and
  * a practical fix: re-calibrating each detector's alarm threshold on a known-clean day
    (here: the 99th percentile of its Monday scores, i.e. a 1% alarm budget) and the
    Thursday detection rate that remains at that threshold.

Run after train_eval.py:   python cross_day.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

import config
import data as D
import models as M


def score_of(name, det, X):
    return det.decision_function(X) if name == config.BASELINE else M.attack_score(det, X)


def main():
    mon = D.load_monday()
    Xm, _, _ = D.xy(mon)
    th = D.load_http()
    X, labels, y = D.xy(th)
    _, te = D.split_indices(labels)
    metrics = json.loads((config.RESULTS_DIR / "metrics.json").read_text())["binary"]
    print(f"Monday (all benign) HTTP flows: {len(Xm):,}")

    names = config.ML_MODELS + [config.BASELINE]
    res, rows = {}, []
    for n in names:
        det = joblib.load(config.MODELS_DIR / f"binary_{n}.joblib")
        pred_m = det.predict(Xm)
        s_m = score_of(n, det, Xm)
        s_t = score_of(n, det, X.iloc[te])
        thr = float(np.quantile(s_m, 0.99))
        # strictly greater than the clean-day 99th percentile keeps Monday alarms <= 1%
        recal_recall = float((s_t[y[te] == 1] > thr).mean())
        recal_fpr_thu = float((s_t[y[te] == 0] > thr).mean())
        r = {"thursday_fpr": metrics[n]["fpr"], "monday_fpr": float(pred_m.mean()),
             "monday_false_alerts": int(pred_m.sum()), "monday_flows": int(len(Xm)),
             "thursday_recall": metrics[n]["recall"],
             "recalibrated_threshold": thr, "recalibrated_monday_fpr": float((s_m > thr).mean()),
             "recalibrated_thursday_recall": recal_recall,
             "recalibrated_thursday_fpr": recal_fpr_thu}
        res[n] = r
        rows.append({"Detector": config.DISPLAY[n], "FPR Thursday (test)": r["thursday_fpr"],
                     "FPR Monday (unseen day)": r["monday_fpr"],
                     "False alerts on Monday": r["monday_false_alerts"],
                     "Recall Thursday (default threshold)": r["thursday_recall"],
                     "Recall Thursday (threshold re-calibrated on Monday)": recal_recall,
                     "FPR Thursday (re-calibrated)": recal_fpr_thu})
        print(f"  {config.DISPLAY[n]:34s} FPR Thu {r['thursday_fpr']:.4f} -> Mon {r['monday_fpr']:.4f} "
              f"({r['monday_false_alerts']:,} false alerts) | recalibrated: recall {recal_recall:.3f}, "
              f"Thu FPR {recal_fpr_thu:.4f}")

    (config.RESULTS_DIR / "cross_day.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(rows).round(4).to_csv(config.TAB_DIR / "table_cross_day.csv", index=False)
    print("Saved results/cross_day.json")


if __name__ == "__main__":
    main()

"""
baseline.py
The conventional detector: a statistical threshold IDS.

Why this baseline?  Conventional web scanners and WAFs detect attacks by matching payload
signatures (e.g. "' OR 1=1", "<script>").  CIC-IDS2017's CSV release contains only
flow statistics -- no payloads -- so signature matching cannot be applied to it at all.
The conventional, non-ML approach that *does* apply to flow records is the classic
statistical anomaly detector (Denning, 1987): profile the normal range of each traffic
metric, and raise an alarm when a flow falls outside it.

How it works (fully transparent, no learning from attacks):
  1. From BENIGN training flows only, learn each feature's normal range
     [0.5th percentile, 99.5th percentile].
  2. Anomaly score of a flow = number of features outside their normal range.
  3. Alarm threshold m = the smallest integer whose false-alarm rate on benign training
     flows is <= the alarm budget (config.TARGET_FPR, 1%), i.e. the way an analyst would
     tune a rule set to an acceptable alert volume.
Every alarm is explainable: explain() lists exactly which metrics were out of range.
"""
from __future__ import annotations

import numpy as np

import config


class ThresholdIDS:
    def __init__(self, q_low=0.005, q_high=0.995, target_fpr=config.TARGET_FPR):
        self.q_low, self.q_high, self.target_fpr = q_low, q_high, target_fpr

    def fit(self, X, y=None):
        """Profile benign traffic. If y is given, attack rows are excluded (never learned)."""
        X = np.asarray(X, dtype=float)
        if y is not None:
            X = X[np.asarray(y) == 0]
        self.lo_ = np.quantile(X, self.q_low, axis=0)
        self.hi_ = np.quantile(X, self.q_high, axis=0)
        s = self.decision_function(X)
        self.m_ = next(m for m in range(1, X.shape[1] + 2) if (s >= m).mean() <= self.target_fpr)
        self.benign_fpr_ = float((s >= self.m_).mean())
        return self

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return ((X < self.lo_) | (X > self.hi_)).sum(axis=1).astype(float)

    def predict(self, X) -> np.ndarray:
        return (self.decision_function(X) >= self.m_).astype(int)

    def explain(self, x, feature_names) -> list[tuple]:
        """Which metrics of a single flow are outside the benign range."""
        x = np.asarray(x, dtype=float)
        out = np.where((x < self.lo_) | (x > self.hi_))[0]
        return [(feature_names[i], float(x[i]), float(self.lo_[i]), float(self.hi_[i])) for i in out]

"""
engineering.py -- learned preprocessing steps for the request-level models.

ProfileFeatures   application profile learned from the *benign training* requests only
                  (anomaly-detection features in the spirit of Kruegel & Vigna's per-parameter
                  models): for every endpoint/parameter it learns the normal value length,
                  character set and character classes, then measures how far a new request
                  departs from that profile. This targets the anomalies that carry no attack
                  keyword (malformed values, parameter tampering, unexpected parameters).
                  It is application-specific by design: profiles describe one application.
CorrelationFilter feature selection: drops a feature whose absolute Pearson correlation with
                  an earlier kept feature exceeds a threshold (fitted on training data).

Both are scikit-learn transformers, so inside a Pipeline they are re-fitted on every
training fold and never see the test data.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from webreq.features import SPECIAL, full_decode

PROFILE_FEATURES = ["prof_new_endpoint", "prof_new_params", "prof_len_z", "prof_new_chars", "prof_new_classes"]
PROFILE_LABELS = {"prof_new_endpoint": "endpoint never seen in normal traffic",
                  "prof_new_params": "parameters never seen at this endpoint",
                  "prof_len_z": "value length vs. normal for this parameter (z-score)",
                  "prof_new_chars": "characters never seen in this parameter",
                  "prof_new_classes": "character classes never seen in this parameter"}


def context(req) -> tuple:
    """What ProfileFeatures needs from a request: (endpoint, [(param name, decoded value)])."""
    return req.endpoint(), [(f"{loc}:{name}", full_decode(v)) for loc, name, v in req.params()]


def _classes(v: str) -> set:
    out = set()
    for ch in v:
        out.add("alpha" if ch.isalpha() and ord(ch) < 128 else "digit" if ch.isdigit() else
                "space" if ch.isspace() else "special" if ch in SPECIAL else
                "nonascii" if ord(ch) > 127 else "punct")
    return out


class ProfileFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, min_count: int = 5, max_z: float = 50.0):
        self.min_count = min_count
        self.max_z = max_z

    def fit(self, X, y=None):
        ctx = list(X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X)
        if y is None:
            benign = np.ones(len(ctx), bool)
        else:
            y = np.asarray(y)
            benign = (y == 0) if np.issubdtype(y.dtype, np.number) else np.isin(y.astype(str), ["0", "norm", "benign"])
        lens, chars, classes = defaultdict(list), defaultdict(set), defaultdict(set)
        self.endpoints_ = set()
        for (ep, params), b in zip(ctx, benign):
            if not b:
                continue
            self.endpoints_.add(ep)
            for name, v in params:
                for key in ((ep, name), name):              # endpoint-specific and global profile
                    lens[key].append(len(v))
                    chars[key].update(v)
                    classes[key].update(_classes(v))
        self.stats_ = {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in lens.items()}
        self.chars_, self.classes_ = dict(chars), dict(classes)
        self.all_chars_ = set().union(*chars.values()) if chars else set()
        return self

    def _key(self, ep, name):
        k = (ep, name)
        if k in self.stats_ and self.stats_[k][2] >= self.min_count:
            return k
        return name if name in self.stats_ else None

    def transform(self, X):
        ctx = list(X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X)
        out = np.zeros((len(ctx), len(PROFILE_FEATURES)))
        for i, (ep, params) in enumerate(ctx):
            new_params = z = new_chars = new_cls = 0.0
            for name, v in params:
                k = self._key(ep, name)
                if k is None:
                    new_params += 1
                    new_chars = max(new_chars, len(set(v) - self.all_chars_))
                    continue
                if (ep, name) not in self.stats_:
                    new_params += 1
                mu, sd, _ = self.stats_[k]
                z = max(z, min(abs(len(v) - mu) / (sd + 1.0), self.max_z))
                new_chars = max(new_chars, len(set(v) - self.chars_[k]))
                new_cls = max(new_cls, len(_classes(v) - self.classes_[k]))
            out[i] = [float(ep not in self.endpoints_), new_params, z, new_chars, new_cls]
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(PROFILE_FEATURES)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        A = np.asarray(X, dtype=float)
        sd = A.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):     # constant columns -> NaN
            C = np.nan_to_num(np.abs(np.corrcoef(A, rowvar=False)), nan=0.0)
        keep = []
        for j in range(A.shape[1]):
            if sd[j] == 0:                                   # constant feature
                continue
            if all(C[j, k] <= self.threshold for k in keep):
                keep.append(j)
        self.keep_ = np.array(keep, dtype=int)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]

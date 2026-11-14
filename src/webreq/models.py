"""
models.py
Request-level AI/ML models. Each takes a DataFrame holding the interpretable features
(features.FEATURES), the decoded request text ("text") and the request context ("ctx",
used by the learned application profile):

  * lr       Logistic Regression on the interpretable features (log-scaled, standardised,
             optional correlation filter; L2 / L1 / elastic-net penalty chosen by tuning)
  * rf       Random Forest on the interpretable features
  * xgb      XGBoost on the interpretable features (regularised: depth, min-child-weight,
             L1/L2 leaf penalties, row/column subsampling)
  * char_svm NLP model: TF-IDF of character 1-4-grams of the decoded request + linear SVM.
             It needs no hand-written features and learns payload fragments directly.
  * hybrid   char n-gram TF-IDF + the interpretable (and profile) features in one linear SVM

profile=True adds ProfileFeatures (webreq/engineering.py), learned from the benign training
requests of the application being protected; use it only when training and test traffic come
from the same application (CSIC 2010), not for the cross-application payload models.
Hyperparameters found by webreq/tuning.py (inner cross-validation on the training split only)
are applied automatically when results/webreq/tuned_params.json exists.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

import config
from models import BalancedXGB, signed_log1p
from webreq import features as F
from webreq.engineering import CorrelationFilter, ProfileFeatures, context  # noqa: F401

NAMES = ["lr", "rf", "xgb", "char_svm", "hybrid"]
DISPLAY = {"lr": "Logistic Regression (features)", "rf": "Random Forest (features)",
           "xgb": "XGBoost (features)", "char_svm": "Char n-gram TF-IDF + SVM (NLP)",
           "hybrid": "Hybrid: char n-grams + features (SVM)",
           "signature": "Signature rules (conventional WAF)"}
SHORT = {"lr": "LogReg", "rf": "RForest", "xgb": "XGBoost", "char_svm": "Char-SVM", "hybrid": "Hybrid",
         "signature": "Signatures"}
TUNED = config.WEB_RESULTS / "tuned_params.json"


def _scaled():
    return [("log", FunctionTransformer(signed_log1p)), ("scale", StandardScaler())]


def _columns(features, profile):
    cols = [("f", "passthrough", features)]
    if profile:
        cols.append(("p", ProfileFeatures(), ["ctx"]))
    return ColumnTransformer(cols)


def _tfidf():
    return TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2, max_features=200_000,
                           sublinear_tf=True, lowercase=True)


def tuned_params() -> dict:
    return json.loads(TUNED.read_text()) if TUNED.exists() else {}


def build(profile: bool = False, features: list | None = None, tuned: bool = True,
          names: list | None = None) -> dict:
    rs = config.RANDOM_STATE
    feats = features or F.FEATURES
    num = [("f", Pipeline(_scaled()), feats)]
    if profile:
        num.append(("p", Pipeline([("prof", ProfileFeatures())] + _scaled()), ["ctx"]))
    models = {
        "lr": Pipeline([("cols", _columns(feats, profile)), *_scaled(), ("sel", "passthrough"),
                        ("clf", LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                                                   random_state=rs))]),
        "rf": Pipeline([("cols", _columns(feats, profile)),
                        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                       n_jobs=-1, random_state=rs))]),
        "xgb": Pipeline([("cols", _columns(feats, profile)),
                         ("clf", BalancedXGB(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=rs))]),
        "char_svm": Pipeline([("cols", ColumnTransformer([("t", _tfidf(), "text")])),
                              ("clf", LinearSVC(C=0.5, class_weight="balanced", max_iter=20000, random_state=rs))]),
        "hybrid": Pipeline([("cols", ColumnTransformer([("t", _tfidf(), "text"), *num])),
                            ("clf", LinearSVC(C=0.5, class_weight="balanced", max_iter=20000, random_state=rs))]),
    }
    if tuned:
        for name, params in tuned_params().items():
            if name in models:
                models[name].set_params(**{k: _restore(v) for k, v in params.items()})
    return {n: models[n] for n in (names or NAMES)}


def _restore(v):
    """JSON stores transformers by name; turn them back into objects."""
    if v == "CorrelationFilter(0.95)":
        return CorrelationFilter(0.95)
    return v


def score(model, X) -> np.ndarray:
    """Continuous attack score (higher = more likely an attack) for binary models."""
    clf = model.steps[-1][1]
    return model.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else model.decision_function(X)


def to_frame(reqs) -> pd.DataFrame:
    X = F.frame(reqs)
    X["text"] = [r.to_text() for r in reqs]
    X["ctx"] = [context(r) for r in reqs]
    return X

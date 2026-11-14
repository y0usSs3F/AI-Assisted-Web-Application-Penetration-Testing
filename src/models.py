"""
models.py
The five supervised AI/ML detectors, applied to CICFlowMeter flow statistics.

  * Logistic Regression  -- linear baseline
  * Linear SVM           -- linear max-margin classifier
  * Random Forest        -- bagged decision trees
  * Gradient-Boosted Trees (HistGradientBoosting) -- boosted trees, the scikit-learn
    implementation of the XGBoost/LightGBM family (XGBoost itself is a drop-in swap)
  * Multi-Layer Perceptron -- feed-forward neural network

Flow statistics (byte counts, durations, inter-arrival times) are heavy-tailed and span
many orders of magnitude, so the linear models and the MLP receive a signed log
transform followed by standardisation; tree models are scale-invariant and use raw values.
The attack classes are rare (about 11% of HTTP flows, and only 21 SQL-injection flows), so
every model is class-balanced: class_weight="balanced" where supported, and balanced
sample weights for the MLP.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

import config


def signed_log1p(X):
    """sign(x) * log(1 + |x|): compresses heavy tails, keeps the sign of -1 sentinels."""
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def _log_scaled(clf) -> Pipeline:
    return Pipeline([
        ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def build_models() -> dict:
    """Fresh, unfitted detectors keyed by short name (same set for binary and multi-class)."""
    rs = config.RANDOM_STATE
    return {
        "logreg": _log_scaled(LogisticRegression(C=1.0, max_iter=5000,
                                                 class_weight="balanced", random_state=rs)),
        "linsvm": _log_scaled(LinearSVC(C=0.5, max_iter=20000,
                                        class_weight="balanced", random_state=rs)),
        "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=rs),
        "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              class_weight="balanced", random_state=rs),
        "mlp": _log_scaled(MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4,
                                         batch_size=256, learning_rate_init=1e-3,
                                         max_iter=300, early_stopping=True,
                                         n_iter_no_change=10, random_state=rs)),
    }


def fit(name: str, model, X, y):
    """Fit a detector, passing balanced sample weights to the MLP (no class_weight option)."""
    if name == "mlp":
        model.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def attack_score(model, X) -> np.ndarray:
    """Continuous attack score for a binary detector (higher = more likely an attack)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

"""
models.py
The five supervised AI/ML detectors, applied to CICFlowMeter flow statistics.

  * Logistic Regression  -- linear baseline
  * Linear SVM           -- linear max-margin classifier
  * Random Forest        -- bagged decision trees
  * Gradient-Boosted Trees (HistGradientBoosting) -- scikit-learn's histogram boosting
  * XGBoost                -- regularised gradient boosting (Chen & Guestrin, KDD 2016)
  * Multi-Layer Perceptron -- feed-forward neural network

Flow statistics (byte counts, durations, inter-arrival times) are heavy-tailed and span
many orders of magnitude, so the linear models and the MLP receive a signed log
transform followed by standardisation; tree models are scale-invariant and use raw values.
The attack classes are rare (about 11% of HTTP flows, and only 21 SQL-injection flows), so
every model is class-balanced: class_weight="balanced" where supported, and balanced
sample weights for the MLP and XGBoost.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config


class BalancedXGB(ClassifierMixin, BaseEstimator):
    """XGBoost with the same interface as the scikit-learn detectors.

    XGBoost needs integer labels 0..k-1 and has no class_weight option, so this wrapper
    encodes string labels, applies balanced sample weights, and decodes predictions.
    """

    def __init__(self, n_estimators=300, learning_rate=0.1, max_depth=6, random_state=None,
                 min_child_weight=1.0, subsample=1.0, colsample_bytree=1.0,
                 reg_lambda=1.0, reg_alpha=0.0, gamma=0.0, n_jobs=-1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        # regularisation (XGBoost defaults; tuned for the request models in webreq/tuning.py)
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        yi = self.le_.transform(y)
        self.model_ = XGBClassifier(n_estimators=self.n_estimators,
                                    learning_rate=self.learning_rate, max_depth=self.max_depth,
                                    min_child_weight=self.min_child_weight, subsample=self.subsample,
                                    colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
                                    reg_alpha=self.reg_alpha, gamma=self.gamma,
                                    tree_method="hist", n_jobs=self.n_jobs, random_state=self.random_state,
                                    eval_metric="logloss")
        self.model_.fit(np.asarray(X, dtype=float), yi,
                        sample_weight=compute_sample_weight("balanced", yi))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.le_.inverse_transform(self.predict_proba(X).argmax(axis=1))

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


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
        "xgb": BalancedXGB(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=rs),
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

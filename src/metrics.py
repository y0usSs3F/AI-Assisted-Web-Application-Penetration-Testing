"""
metrics.py
Evaluation metrics shared by every experiment.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)

import config


def tpr_at_fpr(y_true, score, target: float) -> float:
    """Highest detection rate achievable while keeping false alarms <= target."""
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def binary_metrics(y_true, y_pred, score=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        m["roc_auc"] = roc_auc_score(y_true, score)
        m["pr_auc"] = average_precision_score(y_true, score)
        m["tpr_at_1pct_fpr"] = tpr_at_fpr(y_true, score, 0.01)
        m["tpr_at_0.1pct_fpr"] = tpr_at_fpr(y_true, score, 0.001)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def recall_by_attack(labels, y_pred) -> dict:
    """Detection rate of a binary detector on each attack type separately."""
    labels, y_pred = np.asarray(labels), np.asarray(y_pred)
    return {a: float(y_pred[labels == a].mean()) for a in config.ATTACKS if (labels == a).any()}


def multiclass_metrics(y_true, y_pred) -> dict:
    labels = config.CLASSES
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

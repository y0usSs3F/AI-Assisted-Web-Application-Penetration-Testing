"""
data.py
Load the processed datasets and produce the single, reproducible train/test split.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def features() -> list[str]:
    return json.loads(config.FEATURES_JSON.read_text())


def load_http() -> pd.DataFrame:
    """Thursday HTTP flows (main dataset)."""
    return pd.read_csv(config.THURSDAY_HTTP)


def load_monday() -> pd.DataFrame:
    """Monday HTTP flows (all benign, unseen day)."""
    return pd.read_csv(config.MONDAY_HTTP)


def xy(df: pd.DataFrame):
    """-> X (DataFrame of model features), labels (str array), y_bin (0 benign / 1 attack)."""
    X = df[features()].astype(float)
    labels = df[config.LABEL_COL].to_numpy()
    return X, labels, (labels != config.BENIGN).astype(int)


def split_indices(labels):
    """Stratified on the attack *type*, so every class (even 21 SQLi flows) is in both sets.
    The same split is used for the binary and the multi-class task."""
    idx = np.arange(len(labels))
    return train_test_split(idx, test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_STATE, stratify=labels)

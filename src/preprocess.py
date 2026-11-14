"""
preprocess.py
Turn the raw CIC-IDS2017 CSVs into clean, analysis-ready datasets.

Cleaning steps (each one is counted and written to results/data_profile.json):
  1. strip whitespace from column names; repair the mis-encoded labels
     ("Web Attack � Brute Force" -> "Brute Force", etc.)
  2. drop "Fwd Header Length.1" (an exact duplicate of "Fwd Header Length")
  3. replace +/-inf with NaN and drop those rows (zero-duration flows whose
     Flow Bytes/s and Flow Packets/s are undefined; all of them are benign)
  4. drop exact duplicate rows (prevents identical flows landing in train and test)
  5. drop constant columns (no information)
  6. HTTP scope: keep flows to the web server port (80) and drop "Destination Port".
     All web attacks in CIC-IDS2017 target port 80, so keeping other ports would let a
     model "detect" attacks simply by recognising web traffic (a shortcut). Restricting to
     HTTP flows forces it to separate malicious from benign *web* traffic.

Outputs (data/processed/):
  thursday_all.csv.gz   all Thursday-morning traffic after steps 1-5 (for the shortcut study)
  thursday_http.csv.gz  HTTP flows only after step 6                  (main dataset)
  monday_http.csv.gz    Monday (all benign) HTTP flows, same columns  (unseen-day test)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config

DUP_COLUMN = "Fwd Header Length.1"
PORT = "Destination Port"


def normalize_label(raw) -> str:
    s = str(raw).strip()
    if "Brute Force" in s:
        return "Brute Force"
    if "XSS" in s:
        return "XSS"
    if "Sql Injection" in s or "SQL Injection" in s:
        return "SQL Injection"
    return s.upper() if s.upper() == "BENIGN" else s


def _counts(df: pd.DataFrame) -> dict:
    return {k: int(v) for k, v in df[config.LABEL_COL].value_counts().items()}


def read_clean(path, port: int | None = None, chunksize: int = 200_000):
    """Read a raw CSV in chunks; steps 1-3 (+ optional port filter). Returns (df, stats)."""
    parts, stats = [], {"raw_rows": 0, "raw_labels": {}, "port_rows": 0, "nonfinite_rows": 0}
    for chunk in pd.read_csv(path, encoding="latin-1", low_memory=False, chunksize=chunksize):
        chunk.columns = [c.strip() for c in chunk.columns]
        chunk[config.LABEL_COL] = chunk[config.LABEL_COL].map(normalize_label)
        stats["raw_rows"] += len(chunk)
        for k, v in chunk[config.LABEL_COL].value_counts().items():
            stats["raw_labels"][k] = stats["raw_labels"].get(k, 0) + int(v)
        chunk = chunk.drop(columns=[DUP_COLUMN], errors="ignore")
        feats = [c for c in chunk.columns if c != config.LABEL_COL]
        chunk[feats] = chunk[feats].apply(pd.to_numeric, errors="coerce")
        if port is not None:
            chunk = chunk[chunk[PORT] == port]
            stats["port_rows"] += len(chunk)
        finite = np.isfinite(chunk[feats].to_numpy(dtype=float)).all(axis=1)
        stats["nonfinite_rows"] += int((~finite).sum())
        parts.append(chunk[finite])
    df = pd.concat(parts, ignore_index=True)
    return df, stats


def build() -> dict:
    raw = config.RAW_DIR
    profile = {}

    # ---------------- Thursday (web attacks) ----------------
    th, st = read_clean(raw / config.THURSDAY_FILE)
    after_finite = _counts(th)
    n_dups = int(th.duplicated().sum())
    th = th.drop_duplicates().reset_index(drop=True)
    feats_all = [c for c in th.columns if c != config.LABEL_COL]
    const_all = [c for c in feats_all if th[c].nunique() <= 1]
    th = th.drop(columns=const_all)
    th.to_csv(config.THURSDAY_ALL, index=False, compression="gzip")

    http = th[th[PORT] == config.HTTP_PORT].drop(columns=[PORT]).reset_index(drop=True)
    feats_http = [c for c in http.columns if c != config.LABEL_COL]
    const_http = [c for c in feats_http if http[c].nunique() <= 1]
    http = http.drop(columns=const_http)
    features = [c for c in http.columns if c != config.LABEL_COL]
    http.to_csv(config.THURSDAY_HTTP, index=False, compression="gzip")

    profile["thursday"] = {
        "raw_rows": st["raw_rows"], "raw_labels": st["raw_labels"],
        "nonfinite_rows_removed": st["nonfinite_rows"], "after_nonfinite": after_finite,
        "duplicates_removed": n_dups, "after_dedup": _counts(th), "after_dedup_rows": len(th),
        "constant_columns_removed": const_all,
        "http_rows": len(http), "http_labels": _counts(http),
        "http_extra_constant_columns_removed": const_http,
    }
    print(f"  Thursday: {st['raw_rows']:,} raw -> -{st['nonfinite_rows']} non-finite -> "
          f"-{n_dups:,} duplicates -> {len(th):,} flows -> {len(http):,} HTTP flows")
    print(f"            HTTP labels: {_counts(http)}")

    # ---------------- Monday (benign day) ----------------
    mo, sm = read_clean(raw / config.MONDAY_FILE, port=config.HTTP_PORT)
    mo = mo[features + [config.LABEL_COL]]
    n_dups_m = int(mo.duplicated().sum())
    mo = mo.drop_duplicates().reset_index(drop=True)
    mo.to_csv(config.MONDAY_HTTP, index=False, compression="gzip")
    profile["monday"] = {
        "raw_rows": sm["raw_rows"], "raw_labels": sm["raw_labels"], "http_rows_raw": sm["port_rows"],
        "nonfinite_rows_removed": sm["nonfinite_rows"], "duplicates_removed": n_dups_m,
        "http_rows": len(mo), "http_labels": _counts(mo),
    }
    print(f"  Monday:   {sm['raw_rows']:,} raw -> {sm['port_rows']:,} HTTP -> "
          f"-{sm['nonfinite_rows']} non-finite -> -{n_dups_m:,} duplicates -> {len(mo):,} flows")

    profile["n_features"] = len(features)
    config.FEATURES_JSON.write_text(json.dumps(features, indent=2))
    config.PROFILE_JSON.write_text(json.dumps(profile, indent=2))
    write_feature_medians(http, features)
    print(f"  {len(features)} model features -> {config.FEATURES_JSON.name}")
    return profile


def write_feature_medians(http: pd.DataFrame, features: list[str]) -> None:
    """Median of every feature over benign HTTP flows. Used by the lab harness to fill a
    feature that a particular flow-meter version does not output (and to warn about it)."""
    med = http.loc[http[config.LABEL_COL] == config.BENIGN, features].median()
    (config.RESULTS_DIR / "feature_medians.json").write_text(
        json.dumps({k: float(v) for k, v in med.items()}, indent=2))


if __name__ == "__main__":
    build()

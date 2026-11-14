"""
flow_columns.py
Map flow CSVs produced by different CICFlowMeter versions onto the exact feature names
and units the models were trained on (CIC-IDS2017 "MachineLearningCVE" = CICFlowMeter-V3).

Supported inputs (names taken from each tool's source code):
  * CIC-IDS2017 CSVs themselves (CICFlowMeter-V3 long names, e.g. "Total Fwd Packets")
  * Java CICFlowMeter v4 (github.com/ahlashkari/CICFlowMeter), e.g. "Total Fwd Packet",
    "FWD Init Win Bytes", "Fwd Seg Size Min"                  -- time unit: microseconds
  * CSE-CIC-IDS2018-style abbreviations, e.g. "Tot Fwd Pkts", "Init Fwd Win Byts"
  * Python port `cicflowmeter` (github.com/hieulw/cicflowmeter), e.g. "tot_fwd_pkts",
    "init_fwd_win_byts"                                         -- time unit: SECONDS

Steps: (1) match columns through an alias table (case/punctuation-insensitive);
(2) convert time features to microseconds when the input is in seconds;
(3) derive features that are exact duplicates in CIC-IDS2017 (e.g. "Avg Fwd Segment Size"
equals "Fwd Packet Length Mean"); (4) any feature still missing is filled with its benign
median (results/feature_medians.json) and reported, so you know the score is less reliable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# canonical (CIC-IDS2017) name -> aliases used by other CICFlowMeter versions
ALIASES = {
    "Flow Duration": ["flow_duration"],
    "Total Fwd Packets": ["Total Fwd Packet", "Tot Fwd Pkts", "tot_fwd_pkts"],
    "Total Backward Packets": ["Total Bwd packets", "Tot Bwd Pkts", "tot_bwd_pkts"],
    "Total Length of Fwd Packets": ["Total Length of Fwd Packet", "TotLen Fwd Pkts", "totlen_fwd_pkts"],
    "Total Length of Bwd Packets": ["Total Length of Bwd Packet", "TotLen Bwd Pkts", "totlen_bwd_pkts"],
    "Fwd Packet Length Max": ["Fwd Pkt Len Max", "fwd_pkt_len_max"],
    "Fwd Packet Length Min": ["Fwd Pkt Len Min", "fwd_pkt_len_min"],
    "Fwd Packet Length Mean": ["Fwd Pkt Len Mean", "fwd_pkt_len_mean"],
    "Fwd Packet Length Std": ["Fwd Pkt Len Std", "fwd_pkt_len_std"],
    "Bwd Packet Length Max": ["Bwd Pkt Len Max", "bwd_pkt_len_max"],
    "Bwd Packet Length Min": ["Bwd Pkt Len Min", "bwd_pkt_len_min"],
    "Bwd Packet Length Mean": ["Bwd Pkt Len Mean", "bwd_pkt_len_mean"],
    "Bwd Packet Length Std": ["Bwd Pkt Len Std", "bwd_pkt_len_std"],
    "Flow Bytes/s": ["Flow Byts/s", "flow_byts_s"],
    "Flow Packets/s": ["Flow Pkts/s", "flow_pkts_s"],
    "Flow IAT Mean": ["flow_iat_mean"], "Flow IAT Std": ["flow_iat_std"],
    "Flow IAT Max": ["flow_iat_max"], "Flow IAT Min": ["flow_iat_min"],
    "Fwd IAT Total": ["Fwd IAT Tot", "fwd_iat_tot"], "Fwd IAT Mean": ["fwd_iat_mean"],
    "Fwd IAT Std": ["fwd_iat_std"], "Fwd IAT Max": ["fwd_iat_max"], "Fwd IAT Min": ["fwd_iat_min"],
    "Bwd IAT Total": ["Bwd IAT Tot", "bwd_iat_tot"], "Bwd IAT Mean": ["bwd_iat_mean"],
    "Bwd IAT Std": ["bwd_iat_std"], "Bwd IAT Max": ["bwd_iat_max"], "Bwd IAT Min": ["bwd_iat_min"],
    "Fwd PSH Flags": ["fwd_psh_flags"],
    "Fwd Header Length": ["Fwd Header Len", "fwd_header_len"],
    "Bwd Header Length": ["Bwd Header Len", "bwd_header_len"],
    "Fwd Packets/s": ["Fwd Pkts/s", "fwd_pkts_s"],
    "Bwd Packets/s": ["Bwd Pkts/s", "bwd_pkts_s"],
    "Min Packet Length": ["Packet Length Min", "Pkt Len Min", "pkt_len_min"],
    "Max Packet Length": ["Packet Length Max", "Pkt Len Max", "pkt_len_max"],
    "Packet Length Mean": ["Pkt Len Mean", "pkt_len_mean"],
    "Packet Length Std": ["Pkt Len Std", "pkt_len_std"],
    "Packet Length Variance": ["Pkt Len Var", "pkt_len_var"],
    "FIN Flag Count": ["FIN Flag Cnt", "fin_flag_cnt"],
    "SYN Flag Count": ["SYN Flag Cnt", "syn_flag_cnt"],
    "RST Flag Count": ["RST Flag Cnt", "rst_flag_cnt"],
    "PSH Flag Count": ["PSH Flag Cnt", "psh_flag_cnt"],
    "ACK Flag Count": ["ACK Flag Cnt", "ack_flag_cnt"],
    "URG Flag Count": ["URG Flag Cnt", "urg_flag_cnt"],
    "ECE Flag Count": ["ECE Flag Cnt", "ece_flag_cnt"],
    "Down/Up Ratio": ["down_up_ratio"],
    "Average Packet Size": ["Pkt Size Avg", "pkt_size_avg"],
    "Avg Fwd Segment Size": ["Fwd Segment Size Avg", "Fwd Seg Size Avg", "fwd_seg_size_avg"],
    "Avg Bwd Segment Size": ["Bwd Segment Size Avg", "Bwd Seg Size Avg", "bwd_seg_size_avg"],
    "Subflow Fwd Packets": ["Subflow Fwd Pkts", "subflow_fwd_pkts"],
    "Subflow Fwd Bytes": ["Subflow Fwd Byts", "subflow_fwd_byts"],
    "Subflow Bwd Packets": ["Subflow Bwd Pkts", "subflow_bwd_pkts"],
    "Subflow Bwd Bytes": ["Subflow Bwd Byts", "subflow_bwd_byts"],
    "Init_Win_bytes_forward": ["FWD Init Win Bytes", "Init Fwd Win Byts", "init_fwd_win_byts"],
    "Init_Win_bytes_backward": ["Bwd Init Win Bytes", "Init Bwd Win Byts", "init_bwd_win_byts"],
    "act_data_pkt_fwd": ["Fwd Act Data Pkts", "fwd_act_data_pkts"],
    "min_seg_size_forward": ["Fwd Seg Size Min", "fwd_seg_size_min"],
    "Active Mean": ["active_mean"], "Active Std": ["active_std"],
    "Active Max": ["active_max"], "Active Min": ["active_min"],
    "Idle Mean": ["idle_mean"], "Idle Std": ["idle_std"], "Idle Max": ["idle_max"], "Idle Min": ["idle_min"],
}
# identifying columns kept for reporting (never used as model features)
META = {
    "Src IP": ["Source IP", "src_ip"], "Dst IP": ["Destination IP", "dst_ip"],
    "Src Port": ["Source Port", "src_port"], "Dst Port": ["Destination Port", "dst_port"],
    "Protocol": ["protocol"], "Timestamp": ["timestamp"],
}
# features that are exact (or 99.9%) duplicates in CIC-IDS2017 -> derivable if absent
DERIVE = {
    "Avg Fwd Segment Size": "Fwd Packet Length Mean",
    "Avg Bwd Segment Size": "Bwd Packet Length Mean",
    "Subflow Fwd Packets": "Total Fwd Packets",
    "Subflow Fwd Bytes": "Total Length of Fwd Packets",
    "Subflow Bwd Packets": "Total Backward Packets",
    "Subflow Bwd Bytes": "Total Length of Bwd Packets",
}
TIME_FEATURES = [
    "Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _lookup(table: dict) -> dict:
    lk = {}
    for canon, aliases in table.items():
        for a in [canon] + aliases:
            lk[_key(a)] = canon
    return lk


_FEAT_LK, _META_LK = _lookup(ALIASES), _lookup(META)


def detect_style(columns) -> str:
    keys = {_key(c) for c in columns}
    if "flowduration" in keys and any("_" in str(c) and str(c).islower() for c in columns):
        return "python-cicflowmeter"
    if _key("Total Fwd Packet") in keys or _key("FWD Init Win Bytes") in keys:
        return "java-cicflowmeter-v4"
    if _key("Tot Fwd Pkts") in keys:
        return "cse-cic-ids2018"
    return "cic-ids2017"


def normalize(df: pd.DataFrame, features: list[str], time_unit: str = "auto"):
    """-> (X with exactly `features` in order, metadata DataFrame, report dict)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    style = detect_style(df.columns)
    rename, meta_rename = {}, {}
    for c in df.columns:
        k = _key(c)
        if k in _FEAT_LK and _FEAT_LK[k] not in rename.values():
            rename[c] = _FEAT_LK[k]
        elif k in _META_LK and _META_LK[k] not in meta_rename.values():
            meta_rename[c] = _META_LK[k]
    X = df[list(rename)].rename(columns=rename).apply(pd.to_numeric, errors="coerce")
    meta = df[list(meta_rename)].rename(columns=meta_rename)

    unit = time_unit if time_unit != "auto" else ("s" if style == "python-cicflowmeter" else "us")
    if unit == "s":
        tf = [c for c in TIME_FEATURES if c in X.columns]
        X[tf] = X[tf] * 1e6

    derived, imputed = [], []
    for f in features:
        if f in X.columns:
            continue
        if f in DERIVE and DERIVE[f] in X.columns:
            X[f] = X[DERIVE[f]]
            derived.append(f)
    missing = [f for f in features if f not in X.columns]
    if missing:
        med = json.loads((RESULTS / "feature_medians.json").read_text())
        for f in missing:
            X[f] = med[f]
            imputed.append(f)
    X = X[features].replace([np.inf, -np.inf], np.nan)
    bad = X.isna().any(axis=1)
    X = X.fillna(0.0)
    report = {"style": style, "time_unit": unit, "matched": len(features) - len(derived) - len(imputed),
              "derived": derived, "imputed_with_median": imputed, "rows_with_nonfinite_values": int(bad.sum())}
    return X, meta, report

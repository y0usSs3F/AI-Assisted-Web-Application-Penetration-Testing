"""
datasets.py
Download and load the two request-level datasets.

HTTP CSIC 2010 (Spanish National Research Council): 36,000 normal training requests,
  36,000 normal test requests and 25,065 anomalous test requests sent to an e-commerce web
  application. Anomalies include SQL injection, XSS, buffer overflow, CRLF and server-side
  include attacks, information gathering (e.g. backup files) and malformed parameter values.
HttpParamsDataset (Morzeux, MIT licence): 31,067 HTTP parameter values labelled norm, sqli,
  xss, cmdi or path-traversal, with an official train/test split. Benign values come from
  CSIC 2010; attack values from sqlmap, xssya, Vega and FuzzDB.

Both are fetched from public GitHub repositories pinned to a commit, and their SHA-256
digests are checked.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

import config
from webreq.httpreq import parse_raw, split_raw_stream

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

CSIC_URL = ("https://raw.githubusercontent.com/msudol/Web-Application-Attack-Datasets/"
            "424c6e1c1951570473d9d6c0227eaaac6fdd6d02/OriginalDataSets/csic_2010/{name}")
PARAMS_URL = ("https://raw.githubusercontent.com/Morzeux/HttpParamsDataset/"
              "926670a710283f87c05b554680facf3f9530548c/{name}")
FILES = {
    config.CSIC_DIR: {
        "normalTrafficTraining.txt": "d51de812d9201ef2b173b6ae3e3e740c309047ac85545c06c51d6fb1ddbc1e63",
        "normalTrafficTest.txt": "f05dfc312d5d14fd1ed8371de27a9e4deab3dc09265f5d7f9df2643df8385089",
        "anomalousTrafficTest.txt": "12fa4f0d496ceb859bb2652abf7f0f0ed8c59e1d9ce501b8a9a0ef38a625c046",
    },
    config.HTTPPARAMS_DIR: {
        "payload_train.csv": "dfa6e59c87b2bafc485c9cb14d37e3f86607d180e5a22e7c46dae6d3d1dd6a16",
        "payload_test.csv": "d8015e256ce4499c8ac7a8be8dd25de6487cf018601edf1a03a408a2c8f0e39d",
    },
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def download():
    for folder, files in FILES.items():
        folder.mkdir(parents=True, exist_ok=True)
        tmpl = CSIC_URL if folder == config.CSIC_DIR else PARAMS_URL
        for name, digest in files.items():
            dest = folder / name
            if not dest.exists():
                print(f"  GET {name}")
                req = urllib.request.Request(tmpl.format(name=name),
                                             headers={"User-Agent": "research-downloader/1.0"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    dest.write_bytes(r.read())
            ok = _sha(dest) == digest
            print(f"  [verify] {name}: {'OK' if ok else 'SHA-256 MISMATCH'}")
            if not ok:
                sys.exit(f"{dest} does not match the published file")


# ------------------------------------------------------------------ SR-BH 2020
SRBH_DIR = config.DATA_DIR / "raw" / "srbh2020"
SRBH_FILE = "data_capec_multilabel.csv"
SRBH_URL = "https://dataverse.harvard.edu/api/access/datafile/6319496"   # doi:10.7910/DVN/OGOIXX, CC0
SRBH_MD5 = "173ec515308bdce5aec19cfd5b792596"
SRBH_SIZE = 436_437_661


def download_srbh():
    """SR-BH 2020 (Sureda Riera et al., Computers & Security 2022): 907,814 HTTP requests to a
    WordPress server exposed to the Internet in July 2020, labelled normal or with CAPEC
    attack classes. Resumable (HTTP Range), verified against the published MD5."""
    SRBH_DIR.mkdir(parents=True, exist_ok=True)
    dest = SRBH_DIR / SRBH_FILE
    part = dest.with_suffix(".part")
    if not dest.exists():
        have = part.stat().st_size if part.exists() else 0
        while have < SRBH_SIZE:
            req = urllib.request.Request(SRBH_URL, headers={"User-Agent": "research-downloader/1.0",
                                                            "Range": f"bytes={have}-"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                    if have and r.status != 206:                 # server ignored Range: restart
                        f.truncate(0); have = 0
                    while chunk := r.read(1 << 20):
                        f.write(chunk); have += len(chunk)
                        if have % (20 << 20) < (1 << 20):
                            print(f"  SR-BH 2020: {have >> 20} / {SRBH_SIZE >> 20} MB", flush=True)
            except OSError as e:                                  # network hiccup: resume
                print(f"  interrupted at {have >> 20} MB ({e}); resuming", flush=True)
                have = part.stat().st_size
        part.rename(dest)
    md5 = hashlib.md5()
    with open(dest, "rb") as f:
        while chunk := f.read(1 << 22):
            md5.update(chunk)
    ok = md5.hexdigest() == SRBH_MD5
    print(f"  [verify] {SRBH_FILE}: {'OK' if ok else 'MD5 MISMATCH'}")
    if not ok:
        sys.exit(f"{dest} does not match the published file")


# ------------------------------------------------------------------ CSIC 2010
def load_csic(dedupe: bool = True) -> pd.DataFrame:
    """One row per request: the parsed Request, label (1 = anomalous) and source file."""
    rows = []
    for name, label in [("normalTrafficTraining.txt", 0), ("normalTrafficTest.txt", 0),
                        ("anomalousTrafficTest.txt", 1)]:
        text = (config.CSIC_DIR / name).read_text(encoding="latin-1")
        for raw in split_raw_stream(text):
            req = parse_raw(raw)
            rows.append({"request": req, "label": label, "source": name,
                         "key": f"{req.method} {req.path}?{req.query}#{req.body}"})
    df = pd.DataFrame(rows)
    df.attrs["n_raw"] = len(df)
    if dedupe:
        # identical request content (only the session cookie differs) must not appear in
        # both training and test data; identical content with conflicting labels is dropped
        conflict = df.groupby("key")["label"].nunique()
        bad = set(conflict[conflict > 1].index)
        df.attrs["n_conflicting"] = int(df["key"].isin(bad).sum())
        df = df[~df["key"].isin(bad)].drop_duplicates("key").reset_index(drop=True)
    return df


# ------------------------------------------------------------------ HttpParamsDataset
def load_httpparams() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Official train/test split: payload, attack_type (norm|sqli|xss|cmdi|path-traversal)."""
    out = []
    for name in ("payload_train.csv", "payload_test.csv"):
        d = pd.read_csv(config.HTTPPARAMS_DIR / name, keep_default_na=False)
        d = d[["payload", "attack_type"]].astype(str)
        out.append(d)
    train, test = out
    train = train.drop_duplicates("payload")
    test = test.drop_duplicates("payload")
    test = test[~test["payload"].isin(set(train["payload"]))].reset_index(drop=True)
    return train.reset_index(drop=True), test


if __name__ == "__main__":
    download()
    d = load_csic()
    print(f"CSIC 2010: {d.attrs['n_raw']:,} raw -> {len(d):,} unique requests "
          f"({d.attrs['n_conflicting']} with conflicting labels dropped); "
          f"labels {d['label'].value_counts().to_dict()}")
    tr, te = load_httpparams()
    print(f"HttpParams: train {len(tr):,} {tr.attack_type.value_counts().to_dict()}")
    print(f"            test  {len(te):,} {te.attack_type.value_counts().to_dict()}")

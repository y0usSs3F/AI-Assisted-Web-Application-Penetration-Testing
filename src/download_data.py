"""
download_data.py
Download, verify and preprocess the CIC-IDS2017 web-attack data.

    python download_data.py                  # auto: official source, then mirror
    python download_data.py --source mirror  # force the GitHub mirror
    python download_data.py --force          # re-download even if files exist

What it fetches (CIC-IDS2017 "MachineLearningCVE" CSV release, CICFlowMeter features):
  * Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv  -- web attacks + benign traffic
  * Monday-WorkingHours.pcap_ISCX.csv                       -- a fully benign day, used to
                                                               measure false alarms on unseen data
Sources, tried in order until one succeeds:
  1. Official  -- Canadian Institute for Cybersecurity (UNB), MachineLearningCSV.zip
                  (landing page: https://www.unb.ca/cic/datasets/ids-2017.html)
  2. Mirror    -- a public GitHub repository that stores the same release as a split
                  .tar.xz archive; cloned with a blob-less partial clone pinned to a commit,
                  so only the two archive parts are downloaded (~136 MB).
  3. Manual    -- if both fail, the script explains where to put the files by hand.

Integrity: each file is checked against the published row/label counts (a hard check) and
its SHA-256 is compared with the copy used for the paper (a soft check: warning only).
Only standard-library archive modules are used (zipfile, tarfile, lzma), so it works on
Windows, macOS and Linux. Requires `git` for the mirror source.

Dataset citation (requested by the authors):
  I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, "Toward Generating a New Intrusion
  Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018.
"""
from __future__ import annotations

import argparse
import hashlib
import lzma
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

import config
import preprocess

# --------------------------------------------------------------------------- sources
OFFICIAL_ZIP_URLS = [
    "http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip",
    "http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip",
]
MIRROR_REPO = "https://github.com/jonasyr/network-intrusion-ml-transferability.git"
MIRROR_COMMIT = "e2352f8b74d06218cd937fc6e4abf421424e90e6"
MIRROR_PARTS = [
    "data/raw/cic-ids-2017/full_dataset/MachineLearningCVE.tar.xz.partaa",
    "data/raw/cic-ids-2017/full_dataset/MachineLearningCVE.tar.xz.partab",
]

# --------------------------------------------------------------------------- integrity
EXPECTED_COUNTS = {
    config.THURSDAY_FILE: {"BENIGN": 168186, "Brute Force": 1507, "XSS": 652, "SQL Injection": 21},
    config.MONDAY_FILE: {"BENIGN": 529918},
}
EXPECTED_SHA256 = {
    config.THURSDAY_FILE: "d67066211fb1689c78406f1506f4c44704ecb92088353d5c96d96d6474eb819d",
    config.MONDAY_FILE: "852c4beb34eda186f32561fa79df7a0747e92e1a6535b01270820dd9ffe17f34",
}
NEEDED = [config.THURSDAY_FILE, config.MONDAY_FILE]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path) -> bool:
    """Hard check on row/label counts, soft check on SHA-256."""
    name = path.name
    labels = pd.read_csv(path, encoding="latin-1", usecols=lambda c: c.strip() == "Label",
                         low_memory=False).iloc[:, 0].map(preprocess.normalize_label)
    counts = labels.value_counts().to_dict()
    ok = counts == EXPECTED_COUNTS[name]
    print(f"  [verify] {name}: {len(labels):,} rows, labels {counts} -> "
          f"{'OK' if ok else 'MISMATCH (expected ' + str(EXPECTED_COUNTS[name]) + ')'}")
    digest = sha256(path)
    if digest != EXPECTED_SHA256[name]:
        print(f"  [verify] note: SHA-256 differs from the copy used in the paper "
              f"({digest[:12]}... vs {EXPECTED_SHA256[name][:12]}...). Counts decide validity.")
    return ok


# --------------------------------------------------------------------------- source 1
def _download(url: str, dest: Path) -> bool:
    try:
        print(f"  [official] GET {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "research-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            for chunk in iter(lambda: r.read(1 << 20), b""):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  [official] {done/1e6:7.1f} / {total/1e6:.1f} MB", end="", flush=True)
        print()
        return zipfile.is_zipfile(dest)
    except Exception as e:  # noqa: BLE001
        print(f"  [official] failed: {type(e).__name__}: {e}")
        return False


def from_official(raw_dir: Path) -> bool:
    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "MachineLearningCSV.zip"
        for url in OFFICIAL_ZIP_URLS:
            if _download(url, zpath):
                break
        else:
            return False
        with zipfile.ZipFile(zpath) as zf:
            for name in NEEDED:
                member = next((m for m in zf.namelist() if Path(m).name == name), None)
                if member is None:
                    print(f"  [official] {name} not found in archive")
                    return False
                with zf.open(member) as src, open(raw_dir / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  [official] extracted {name}")
    return True


# --------------------------------------------------------------------------- source 2
def _git(*args, cwd=None, timeout=900):
    subprocess.run(["git", *args], cwd=cwd, check=True, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def from_mirror(raw_dir: Path) -> bool:
    if shutil.which("git") is None:
        print("  [mirror] git is not installed")
        return False
    try:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "mirror"
            print(f"  [mirror] partial clone of {MIRROR_REPO} (metadata only)")
            _git("clone", "--quiet", "--filter=blob:none", "--no-checkout", MIRROR_REPO, str(repo))
            print("  [mirror] fetching the two archive parts (~136 MB) at commit "
                  f"{MIRROR_COMMIT[:10]}")
            _git("checkout", "--quiet", MIRROR_COMMIT, "--", *MIRROR_PARTS, cwd=repo)
            archive = Path(td) / "MachineLearningCVE.tar.xz"
            with open(archive, "wb") as out:
                for part in MIRROR_PARTS:
                    with open(repo / part, "rb") as src:
                        shutil.copyfileobj(src, out)
            print("  [mirror] extracting needed files from the archive")
            with tarfile.open(archive, mode="r:xz") as tar:
                members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
                for name in NEEDED:
                    if name in members:                        # stored as plain CSV
                        src = tar.extractfile(members[name])
                    elif name + ".xz" in members:              # stored xz-compressed inside
                        src = lzma.open(tar.extractfile(members[name + ".xz"]))
                    else:
                        print(f"  [mirror] {name} not found in archive")
                        return False
                    with src, open(raw_dir / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"  [mirror] extracted {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [mirror] failed: {type(e).__name__}: {e}")
        return False


# --------------------------------------------------------------------------- main
MANUAL_HELP = f"""
Automatic download failed. Download it manually:
  1. Open https://www.unb.ca/cic/datasets/ids-2017.html -> "Download this dataset"
     (http://cicresearch.ca/CICDataset/CIC-IDS-2017/) and get MachineLearningCSV.zip
     (Kaggle also mirrors the same CSVs).
  2. From the zip's MachineLearningCVE/ folder, copy:
        {config.THURSDAY_FILE}
        {config.MONDAY_FILE}
     into: {config.RAW_DIR}
  3. Re-run: python download_data.py
"""


def main():
    ap = argparse.ArgumentParser(description="Download, verify and preprocess CIC-IDS2017.")
    ap.add_argument("--source", choices=["auto", "official", "mirror"], default="auto")
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    ap.add_argument("--skip-preprocess", action="store_true")
    args = ap.parse_args()

    raw = config.RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    have = all((raw / n).exists() for n in NEEDED)
    if have and not args.force:
        print(f"Raw files already present in {raw}")
    else:
        sources = {"auto": [from_official, from_mirror], "official": [from_official],
                   "mirror": [from_mirror]}[args.source]
        for fetch in sources:
            print(f"\n== source: {fetch.__name__.replace('from_', '')} ==")
            if fetch(raw) and all((raw / n).exists() for n in NEEDED):
                break
        else:
            print(MANUAL_HELP)
            sys.exit(1)

    print("\n== verifying ==")
    if not all(verify(raw / n) for n in NEEDED):
        print("Verification failed: the files do not match the published CIC-IDS2017 counts.")
        sys.exit(2)

    if not args.skip_preprocess:
        print("\n== preprocessing ==")
        preprocess.build()
    print("\nDone.")


if __name__ == "__main__":
    main()

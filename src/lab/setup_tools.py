"""
setup_tools.py -- download the practical lab's tools into lab/tools/ (no installation,
no administrator rights, no Docker).

    python -m lab.setup_tools          (run from src/)

  * OWASP Juice Shop 20.3.0  -- the intentionally vulnerable web application (needs Node.js 26,
                                which must already be installed; `node --version`)
  * Eclipse Temurin JRE 21    -- Java runtime for ZAP (portable, unpacked locally)
  * OWASP ZAP 2.17.0          -- the conventional automated vulnerability scanner

Every archive is checked against the checksum its publisher lists before it is unpacked.
Windows x64 and Linux x64 are supported.
"""
from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import config

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

TOOLS = config.LAB_DIR / "tools"
JS_VER, ZAP_VER, JRE_VER = "20.3.0", "2.17.0", "21.0.13_9"
JS_BASE = f"https://github.com/juice-shop/juice-shop/releases/download/v{JS_VER}/"
JRE_BASE = "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.13%2B9/"

if sys.platform == "win32":
    ARTIFACTS = {
        "juice-shop": (JS_BASE + f"juice-shop-{JS_VER}_node26_win32_x64.zip", "md5", "d5172c0cc86b9f476a8d8e2daa7b1d71"),
        "jre": (JRE_BASE + f"OpenJDK21U-jre_x64_windows_hotspot_{JRE_VER}.zip", "sha256",
                "2332c0b4cff6bed722da3567c91d5e47568bcba863fa4945d0a8d6fc76158628"),
    }
else:
    ARTIFACTS = {
        "juice-shop": (JS_BASE + f"juice-shop-{JS_VER}_node26_linux_x64.tgz", "md5", "273d114c0b04d677e6da606beb4a0b81"),
        "jre": (JRE_BASE + f"OpenJDK21U-jre_x64_linux_hotspot_{JRE_VER}.tar.gz", "sha256",
                "a535a1df9c6284d9618872984e026ae0352d1f7e9e0605425c0e6114f47351dd"),
    }
ARTIFACTS["zap"] = (f"https://github.com/zaproxy/zaproxy/releases/download/v{ZAP_VER}/ZAP_{ZAP_VER}_Crossplatform.zip",
                    "sha256", "94c8f767b1c2e94f0db66b3ae56514d5e3f5a728ee1b6c798e0c8fe2d61fbff0")


def _download(url: str, dest: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "lab-setup/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unpack(archive: Path, target: Path):
    tmp = target.with_name(target.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for m in z.namelist():                      # refuse path traversal in archives
                if Path(m).is_absolute() or ".." in Path(m).parts:
                    sys.exit(f"unsafe path in {archive.name}: {m}")
            z.extractall(tmp)
    else:
        with tarfile.open(archive) as t:
            t.extractall(tmp, filter="data")
    inner = list(tmp.iterdir())
    src = inner[0] if len(inner) == 1 and inner[0].is_dir() else tmp
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(src), str(target))
    shutil.rmtree(tmp, ignore_errors=True)


def paths() -> dict:
    """Locations of the unpacked tools (used by run_lab)."""
    exe = ".exe" if sys.platform == "win32" else ""
    zap_script = "zap.bat" if sys.platform == "win32" else "zap.sh"
    return {"juice_shop": TOOLS / "juice-shop", "java": TOOLS / "jre" / "bin" / f"java{exe}",
            "zap_dir": TOOLS / "zap", "zap": TOOLS / "zap" / zap_script,
            "zap_jar": TOOLS / "zap" / f"zap-{ZAP_VER}.jar"}


def main():
    if platform.machine().lower() not in ("amd64", "x86_64"):
        sys.exit("Only x64 machines are supported by the pinned tool builds.")
    node = shutil.which("node")
    if not node:
        sys.exit("Node.js is required for Juice Shop: install Node.js 26 (https://nodejs.org).")
    ver = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
    print(f"Node.js {ver} found")
    if not ver.startswith("v26."):
        print("  warning: this Juice Shop build targets Node.js 26; other versions may fail")
    TOOLS.mkdir(parents=True, exist_ok=True)
    dl = TOOLS / "_downloads"
    dl.mkdir(exist_ok=True)
    for name, (url, algo, expected) in ARTIFACTS.items():
        target = TOOLS / name
        if target.exists():
            print(f"[{name}] already unpacked in {target}")
            continue
        archive = dl / url.rsplit("/", 1)[1]
        if not archive.exists() or _digest(archive, algo) != expected:
            print(f"[{name}] downloading {url}")
            _download(url, archive)
        got = _digest(archive, algo)
        if got != expected:
            sys.exit(f"[{name}] {algo} mismatch: expected {expected}, got {got}")
        print(f"[{name}] {algo} verified; unpacking")
        _unpack(archive, target)
    p = paths()
    missing = [k for k in ("java", "zap_jar") if not p[k].exists()]
    if missing:
        sys.exit(f"unpacked tools are incomplete: {missing}")
    print(f"\nLab tools ready in {TOOLS}")


if __name__ == "__main__":
    main()

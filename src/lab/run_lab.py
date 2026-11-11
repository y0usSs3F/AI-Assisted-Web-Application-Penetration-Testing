"""
run_lab.py -- the practical component, end to end, on one machine (run from src/):

    python -m lab.setup_tools        # once: Juice Shop, Java runtime, OWASP ZAP
    python -m lab.run_lab            # ~45-70 min: benign phase + ZAP scan, then the comparison

  1. starts OWASP Juice Shop (intentionally vulnerable web application) on 127.0.0.1:3000
  2. starts the logging proxy on 127.0.0.1:3001 in front of it
  3. phase A ("benign"): simulated shoppers browse the shop through the proxy
  4. phase B ("zap"):    OWASP ZAP explores the app (requestor + spider), passively scans it
                         and runs a full active scan through the proxy, with every active-scan
                         request tagged by its scan-rule id; ZAP writes its own report
  5. runs lab.compare: the AI prototype scores every logged request and its findings are
     compared with ZAP's.
Authorised, isolated use only: everything binds to localhost, and the only target is the
local Juice Shop instance this script starts and stops.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import config
from lab import benign, compare
from lab.proxy import LabProxy
from lab.setup_tools import paths

APP_PORT, PROXY_PORT, ZAP_PORT = 3000, 3001, 8090
TARGET = f"http://127.0.0.1:{PROXY_PORT}"


def wait_http(url, timeout=240) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return False


def seed_requests() -> list[dict]:
    """Requests that show ZAP the app's API (a tester's usual 'explore' step)."""
    j = ["Content-Type: application/json"]
    return [
        {"url": f"{TARGET}/", "method": "GET"},
        {"url": f"{TARGET}/rest/products/search?q=apple", "method": "GET"},
        {"url": f"{TARGET}/api/Products/1?d=Mon", "method": "GET"},
        {"url": f"{TARGET}/rest/products/1/reviews", "method": "GET"},
        {"url": f"{TARGET}/rest/user/login", "method": "POST", "headers": j,
         "data": '{"email":"shopper@example.com","password":"Secret-123"}'},
        {"url": f"{TARGET}/api/Users/", "method": "POST", "headers": j,
         "data": '{"email":"new@example.com","password":"Secret-123","passwordRepeat":"Secret-123",'
                 '"securityQuestion":{"id":1},"securityAnswer":"blue"}'},
        {"url": f"{TARGET}/api/Feedbacks/", "method": "POST", "headers": j,
         "data": '{"captchaId":1,"captcha":"5","comment":"Nice shop","rating":4}'},
        {"url": f"{TARGET}/rest/track-order/5267-f9cd5882f54c75a3", "method": "GET"},
        {"url": f"{TARGET}/rest/captcha/", "method": "GET"},
        {"url": f"{TARGET}/rest/user/whoami", "method": "GET"},
        {"url": f"{TARGET}/rest/memories/", "method": "GET"},
        {"url": f"{TARGET}/rest/languages", "method": "GET"},
        {"url": f"{TARGET}/api/Quantitys/", "method": "GET"},
        {"url": f"{TARGET}/api/SecurityQuestions/", "method": "GET"},
        {"url": f"{TARGET}/rest/admin/application-configuration", "method": "GET"},
        {"url": f"{TARGET}/redirect?to=https://github.com/juice-shop/juice-shop", "method": "GET"},
        {"url": f"{TARGET}/ftp/", "method": "GET"},
        {"url": f"{TARGET}/rest/chatbot/status", "method": "GET"},
    ]


STATIC = (".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".woff", ".woff2",
          ".ttf", ".map")


def seeds_from_browsing(log_path: Path) -> list[dict]:
    """One request per API endpoint seen during the benign phase, as a tester would hand the
    proxy history to the scanner: method, URL, JSON body and the session's bearer token."""
    from webreq.httpreq import Request
    seen, out = set(), []
    for line in open(log_path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("phase") != "benign" or r["path"].lower().endswith(STATIC) or r["path"].startswith("/socket.io"):
            continue
        key = Request.from_dict(r).endpoint()
        if key in seen or r.get("status", 500) >= 400:
            continue
        seen.add(key)
        h = {k.lower(): v for k, v in r["headers"].items()}
        e = {"url": f"{TARGET}{r['path']}" + (f"?{r['query']}" if r["query"] else ""), "method": r["method"]}
        hdrs = [f"Content-Type: {h['content-type']}"] if "content-type" in h else []
        if "authorization" in h:
            hdrs.append(f"Authorization: {h['authorization']}")
        if hdrs:
            e["headers"] = hdrs
        if r.get("body"):
            e["data"] = r["body"]
        out.append(e)
    return out


def zap_plan(run_dir: Path, scan_minutes: int) -> Path:
    plan = {
        "env": {"contexts": [{"name": "juice-shop", "urls": [TARGET],
                              "includePaths": [f"{TARGET}.*"],
                              "excludePaths": [f"{TARGET}/socket.io.*"]}],
                "parameters": {"failOnError": False, "progressToStdout": True}},
        "jobs": [
            {"type": "requestor", "requests": seed_requests() + seeds_from_browsing(run_dir / "requests.jsonl")},
            {"type": "spider", "parameters": {"context": "juice-shop", "maxDuration": 3, "maxDepth": 6}},
            {"type": "passiveScan-wait", "parameters": {"maxDuration": 5}},
            {"type": "activeScan", "parameters": {
                "context": "juice-shop", "maxScanDurationInMins": scan_minutes,
                "maxRuleDurationInMins": 4, "threadPerHost": 4,
                "injectPluginIdInHeader": True, "handleAntiCSRFTokens": True}},
            {"type": "passiveScan-wait", "parameters": {"maxDuration": 5}},
            {"type": "report", "parameters": {"template": "traditional-json",
                                              "reportDir": str(run_dir), "reportFile": "zap_report"}},
            {"type": "report", "parameters": {"template": "traditional-html",
                                              "reportDir": str(run_dir), "reportFile": "zap_report"}},
        ],
    }
    p = run_dir / "zap_plan.yaml"
    p.write_text(json.dumps(plan, indent=2))       # JSON is valid YAML
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=60, help="benign shopper sessions")
    ap.add_argument("--scan-minutes", type=int, default=40, help="ZAP active-scan time limit")
    ap.add_argument("--skip-zap", action="store_true")
    args = ap.parse_args()

    p = paths()
    if not (p["java"].exists() and p["zap_jar"].exists() and p["juice_shop"].exists()):
        sys.exit("Lab tools missing: run  python -m lab.setup_tools  first.")
    run_dir = config.LAB_DIR / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    print(f"Run directory: {run_dir}")

    print("[1/5] starting OWASP Juice Shop on 127.0.0.1:3000 ...")
    env = os.environ | {"PORT": str(APP_PORT), "NODE_ENV": "production"}
    js_log = open(run_dir / "juice_shop.log", "w")
    shop = subprocess.Popen(["node", "build/app.js"], cwd=p["juice_shop"], env=env,
                            stdout=js_log, stderr=subprocess.STDOUT)
    proxy = None
    try:
        if not wait_http(f"http://127.0.0.1:{APP_PORT}/rest/admin/application-version"):
            sys.exit("Juice Shop did not start; see juice_shop.log")
        print("[2/5] starting the logging proxy on 127.0.0.1:3001")
        proxy = LabProxy(APP_PORT, PROXY_PORT, run_dir / "requests.jsonl").start()

        print(f"[3/5] phase A: {args.sessions} benign shopper sessions")
        proxy.phase = "benign"
        n = benign.run(TARGET, args.sessions)
        print(f"      {n} benign requests")

        if not args.skip_zap:
            print(f"[4/5] phase B: OWASP ZAP explore + active scan (limit {args.scan_minutes} min)")
            proxy.phase = "zap"
            plan = zap_plan(run_dir, args.scan_minutes)
            cmd = [str(p["java"]), "-Xmx2g", "-jar", str(p["zap_jar"]), "-cmd",
                   "-dir", str(run_dir / "zap_home"), "-port", str(ZAP_PORT), "-autorun", str(plan),
                   "-config", "start.checkForUpdates=false", "-config", "start.addonDirectories=",
                   "-config", "database.recoverylog=false"]
            with open(run_dir / "zap.log", "w") as zl:
                rc = subprocess.run(cmd, cwd=p["zap_dir"], stdout=zl, stderr=subprocess.STDOUT,
                                    timeout=(args.scan_minutes + 30) * 60).returncode
            print(f"      ZAP finished (exit {rc}); {proxy.count} requests logged in total")
        proxy.phase = "done"
    finally:
        if proxy:
            proxy.stop()
        shop.terminate()
        try:
            shop.wait(timeout=20)
        except subprocess.TimeoutExpired:
            shop.kill()
        js_log.close()

    print("[5/5] comparing the AI prototype with ZAP")
    compare.run(run_dir)


if __name__ == "__main__":
    main()

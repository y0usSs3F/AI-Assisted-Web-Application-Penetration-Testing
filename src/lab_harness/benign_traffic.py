"""
benign_traffic.py
Generate NORMAL user traffic against an OWASP Juice Shop instance in your lab, so the
AI detectors' false-positive rate can be measured on benign browsing (phase A).

  python benign_traffic.py --target http://localhost:3000 --minutes 5

It behaves like a shopper: opens pages, searches for products, views product details,
registers an account, logs in and checks the basket, with human-like pauses.
Standard library only. Use ONLY against an application you run yourself.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request

WORDS = ["apple", "banana", "juice", "lemon", "orange", "melon", "water", "egg", "carrot", "t-shirt",
         "mug", "sticker", "coaster", "hoodie", "card"]


def call(base, path, method="GET", body=None, token=None):
    req = urllib.request.Request(base + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Firefox/128.0",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception:  # noqa: BLE001
        return None, b""


def session(base, rng):
    """One shopping session."""
    actions = 0
    for path in ["/", "/rest/admin/application-version", "/api/Quantitys/", "/rest/languages"]:
        call(base, path); actions += 1
    for _ in range(rng.randint(2, 5)):
        call(base, f"/rest/products/search?q={rng.choice(WORDS)}"); actions += 1
        call(base, f"/api/Products/{rng.randint(1, 30)}"); actions += 1
        call(base, f"/rest/products/{rng.randint(1, 30)}/reviews"); actions += 1
        time.sleep(rng.uniform(0.5, 3.0))
    email = f"user{rng.randint(10**6, 10**7)}@example.com"
    pwd = "Benign-User-123!"
    call(base, "/api/Users/", "POST", {"email": email, "password": pwd, "passwordRepeat": pwd,
                                        "securityQuestion": {"id": 1}, "securityAnswer": "blue"}); actions += 1
    status, body = call(base, "/rest/user/login", "POST", {"email": email, "password": pwd}); actions += 1
    if status == 200:
        try:
            auth = json.loads(body)["authentication"]
            token, bid = auth["token"], auth.get("bid")
            call(base, "/rest/user/whoami", token=token); actions += 1
            if bid:
                call(base, f"/rest/basket/{bid}", token=token); actions += 1
        except (KeyError, ValueError):
            pass
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="http://localhost:3000")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    end, n_sessions, n_requests = time.time() + 60 * args.minutes, 0, 0
    while time.time() < end:
        n_requests += session(args.target.rstrip("/"), rng)
        n_sessions += 1
        time.sleep(rng.uniform(1, 5))
    print(f"Benign traffic done: {n_sessions} sessions, {n_requests} requests.")


if __name__ == "__main__":
    main()

"""
benign.py -- simulated legitimate shoppers for OWASP Juice Shop (lab phase A).

Each session behaves like a browser user: loads the single-page app and its assets, searches,
opens products and reviews, registers, logs in, fills the basket, adds an address, leaves
feedback (solving the shop's captcha) and tracks an order. About one input in eight is
benign but "awkward" (apostrophes, ampersands, emoticons, percentages), the realistic
content that makes signature rules raise false alarms.
Use only against an application you run yourself.
"""
from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request

UAS = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36",
       "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
       "Mozilla/5.0 (X11; Linux x86_64; rv:144.0) Gecko/20100101 Firefox/144.0",
       "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1"]
WORDS = ["apple", "banana", "juice", "lemon", "orange", "melon", "water", "egg", "carrot", "t-shirt",
         "mug", "sticker", "coaster", "hoodie", "card", "green smoothie", "fruit press", "lemon juice",
         "owasp", "raspberry", "strawberry", "cookie", "eggfruit", "permafrost", "quince"]
AWKWARD_SEARCH = ["o'neill juice", "apple & banana", "50% off", "kid's mug", "rock 'n' roll", "juice (500ml)",
                  "lemon/lime", "best juice!!!", "\"apple juice\"", "café", "100% organic", "mug - 2 pack"]
COMMENTS = ["Great taste, will buy again.", "Delivery was quick.", "A bit too sweet for me.",
            "Love the packaging!", "Good value for money.", "The website is easy to use.",
            "Could you add more vegan options?", "Five stars, my kids love it."]
AWKWARD_COMMENTS = ["Best juice ever <3", "It's OK, but the mug's handle broke :(", "Tasty & fresh -- recommended!",
                    "Price/quality = excellent", "Ordered 2x; both arrived; 10/10", "Wasn't sure at first... now I'm a fan!",
                    "Mmm, \"organic\" & delicious", "a+b=perfect breakfast"]
NAMES = ["Anna Schmidt", "Omar Haddad", "Li Wei", "Sara O'Connor", "John Smith", "Maria Garcia", "Yuki Tanaka",
         "Ahmed Saleh", "Chloé Dubois", "D'Angelo Rossi"]
CITIES = ["Berlin", "Cairo", "Toronto", "Madrid", "Osaka", "Lyon", "Boston", "Dublin"]


class Client:
    def __init__(self, base, rng):
        self.base, self.rng = base, rng
        self.ua = rng.choice(UAS)
        self.token = None
        self.n = 0

    def call(self, path, method="GET", body=None, accept="application/json, text/plain, */*"):
        hdrs = {"User-Agent": self.ua, "Accept": accept, "Accept-Language": "en-US,en;q=0.9"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            hdrs["Content-Type"] = "application/json"
        if self.token:
            hdrs["Authorization"] = f"Bearer {self.token}"
            hdrs["Cookie"] = f"language=en; welcomebanner_status=dismiss; token={self.token}"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=hdrs)
        self.n += 1
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                try:
                    return r.status, json.loads(raw)
                except ValueError:
                    return r.status, raw.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, None
        except Exception:  # noqa: BLE001
            return None, None


def _q(s):
    from urllib.parse import quote
    return quote(s, safe="")


def session(base, rng: random.Random) -> int:
    c = Client(base, rng)
    status, index = c.call("/", accept="text/html")
    if isinstance(index, str):                                  # the SPA's static assets
        for asset in re.findall(r'(?:src|href)="([^"]+\.(?:js|css|ico|png))"', index)[:8]:
            c.call("/" + asset.lstrip("/"), accept="*/*")
    for p in ["/rest/admin/application-version", "/rest/admin/application-configuration",
              "/rest/languages", "/api/Quantitys/", "/rest/products/search?q="]:
        c.call(p)
    products = []
    for _ in range(rng.randint(2, 6)):
        term = rng.choice(AWKWARD_SEARCH) if rng.random() < 0.125 else rng.choice(WORDS)
        st, res = c.call(f"/rest/products/search?q={_q(term)}")
        if isinstance(res, dict):
            products += [p for p in res.get("data", []) if isinstance(p, dict)]
    ids = [p["id"] for p in products if "id" in p] or list(range(1, 25))
    for pid in rng.sample(ids, min(len(ids), rng.randint(1, 4))):
        c.call(f"/api/Products/{pid}?d={_q('Mon Nov 10 2026')}")
        c.call(f"/rest/products/{pid}/reviews")
    for p in products[:2]:
        if p.get("image"):
            c.call(f"/assets/public/images/products/{_q(p['image'])}", accept="image/*")

    if rng.random() < 0.8:                                      # register and log in
        first = rng.choice(NAMES).split()[0].lower().replace("'", "")
        email = f"{first}.{rng.randint(1000, 99999)}@example.com"
        pwd = f"Juice-{rng.randint(10**5, 10**6)}!"
        c.call("/api/SecurityQuestions/")
        c.call("/api/Users/", "POST", {"email": email, "password": pwd, "passwordRepeat": pwd,
                                        "securityQuestion": {"id": rng.randint(1, 8)},
                                        "securityAnswer": rng.choice(["blue", "Rex", "Smith", "Cairo"])})
        st, res = c.call("/rest/user/login", "POST", {"email": email, "password": pwd})
        if st == 200 and isinstance(res, dict):
            auth = res.get("authentication", {})
            c.token, bid = auth.get("token"), auth.get("bid")
            c.call("/rest/user/whoami")
            if bid:
                c.call(f"/rest/basket/{bid}")
                items = []
                for pid in rng.sample(ids, min(len(ids), rng.randint(1, 3))):
                    st, res = c.call("/api/BasketItems/", "POST",
                                     {"ProductId": pid, "BasketId": str(bid), "quantity": 1})
                    if isinstance(res, dict) and isinstance(res.get("data"), dict):
                        items.append(res["data"].get("id"))
                for it in items[:1]:
                    if it:
                        c.call(f"/api/BasketItems/{it}", "PUT", {"quantity": rng.randint(2, 4)})
                c.call(f"/rest/basket/{bid}")
            if rng.random() < 0.5:
                name = rng.choice(NAMES)
                c.call("/api/Addresss/", "POST", {"country": "Germany", "fullName": name,
                                                   "mobileNum": rng.randint(10**8, 10**9),
                                                   "zipCode": str(rng.randint(10000, 99999)),
                                                   "streetAddress": f"{rng.randint(1, 200)} Main St.",
                                                   "city": rng.choice(CITIES), "state": ""})
            c.call("/rest/wallet/balance")
            c.call("/rest/deluxe-membership")
            if rng.random() < 0.6:
                pid = rng.choice(ids)
                msg = rng.choice(AWKWARD_COMMENTS) if rng.random() < 0.125 else rng.choice(COMMENTS)
                c.call(f"/rest/products/{pid}/reviews", "PUT", {"message": msg, "author": email})
    if rng.random() < 0.5:                                      # feedback with captcha
        st, cap = c.call("/rest/captcha/")
        if isinstance(cap, dict) and "captcha" in cap:
            expr = str(cap["captcha"])                          # e.g. "3*5+1" from the shop
            answer = "0"
            if re.fullmatch(r"[0-9+\-* ]{1,40}", expr):         # plain arithmetic only
                answer = str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
            msg = rng.choice(AWKWARD_COMMENTS) if rng.random() < 0.125 else rng.choice(COMMENTS)
            c.call("/api/Feedbacks/", "POST", {"captchaId": cap.get("captchaId"), "captcha": answer,
                                                "comment": msg, "rating": rng.randint(3, 5)})
    if rng.random() < 0.3:
        c.call(f"/rest/track-order/{rng.randint(1000, 9999):x}-{rng.randint(10**15, 10**16):x}")
    return c.n


def run(base: str, sessions: int, seed: int = 42) -> int:
    rng = random.Random(seed)
    total = 0
    for i in range(sessions):
        total += session(base, rng)
        if (i + 1) % 10 == 0:
            print(f"    benign sessions {i + 1}/{sessions}, {total} requests")
    return total

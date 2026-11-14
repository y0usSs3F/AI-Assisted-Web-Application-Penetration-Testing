#!/usr/bin/env bash
# capture_and_scan.sh -- the practical lab session (run on a Linux VM you control):
#   phase A: benign browsing of OWASP Juice Shop          -> benign.pcap
#   phase B: OWASP ZAP full active scan of the same app   -> attack.pcap + zap_report.json/html
#
# Usage:   sudo bash capture_and_scan.sh [OUTDIR] [BENIGN_MINUTES]
# Example: sudo bash capture_and_scan.sh lab_run 5
# Needs:   docker, tcpdump, curl, python3.
# AUTHORISED, ISOLATED LAB ONLY: this launches an intentionally vulnerable app and attacks it.
set -euo pipefail

OUT="${1:-lab_run}"
BENIGN_MIN="${2:-5}"
PORT="${PORT:-3000}"
TARGET="http://localhost:${PORT}"
IFACE="${IFACE:-lo}"            # loopback: ZAP (host network) -> docker-proxy -> Juice Shop
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"

echo "[1/5] starting OWASP Juice Shop on ${TARGET}"
docker rm -f juiceshop >/dev/null 2>&1 || true
docker run -d --rm --name juiceshop -p "${PORT}:3000" bkimminich/juice-shop >/dev/null
until curl -fs "${TARGET}" >/dev/null; do sleep 2; done

echo "[2/5] phase A: benign browsing for ${BENIGN_MIN} min -> ${OUT}/benign.pcap"
tcpdump -i "$IFACE" -s 0 -w "$OUT/benign.pcap" "tcp port ${PORT}" >/dev/null 2>&1 &
TCPDUMP=$!; sleep 2
python3 "$HERE/benign_traffic.py" --target "$TARGET" --minutes "$BENIGN_MIN"
sleep 3; kill "$TCPDUMP"; wait "$TCPDUMP" 2>/dev/null || true

echo "[3/5] phase B: OWASP ZAP full active scan -> ${OUT}/attack.pcap, zap_report.json"
tcpdump -i "$IFACE" -s 0 -w "$OUT/attack.pcap" "tcp port ${PORT}" >/dev/null 2>&1 &
TCPDUMP=$!; sleep 2
chmod 777 "$OUT"
docker run --rm --network host -v "$OUT:/zap/wrk:rw" ghcr.io/zaproxy/zaproxy:stable \
    zap-full-scan.py -t "$TARGET" -m 5 -J zap_report.json -r zap_report.html -I || true
sleep 3; kill "$TCPDUMP"; wait "$TCPDUMP" 2>/dev/null || true

echo "[4/5] stopping Juice Shop"
docker stop juiceshop >/dev/null 2>&1 || true

echo "[5/5] done -- captures and ZAP reports are in ${OUT}"
ls -la "$OUT"
cat <<EOF

Next steps (see README_LAB.md):
  1. Convert both captures to flows with CICFlowMeter:
       ${OUT}/benign.pcap -> ${OUT}/benign_flows.csv
       ${OUT}/attack.pcap -> ${OUT}/attack_flows.csv
  2. Score them:
       python3 ${HERE}/score_flows.py ${OUT}/benign_flows.csv --port ${PORT} --phase benign --out ${OUT}/benign_scored.csv
       python3 ${HERE}/score_flows.py ${OUT}/attack_flows.csv --port ${PORT} --phase attack --out ${OUT}/attack_scored.csv
  3. Compare with ZAP:
       python3 ${HERE}/compare_with_zap.py --zap-json ${OUT}/zap_report.json \\
           --benign ${OUT}/benign_scored.summary.json --attack ${OUT}/attack_scored.summary.json \\
           --out ${OUT}/comparison.md
EOF

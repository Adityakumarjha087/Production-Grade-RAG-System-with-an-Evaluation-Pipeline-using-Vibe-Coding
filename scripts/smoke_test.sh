#!/usr/bin/env bash
# End-to-end smoke test against a running API (default: the docker compose stack).
# Usage: scripts/smoke_test.sh [base_url]
set -euo pipefail
BASE="${1:-http://localhost:8000}"
Q='{"query": "How long must intermediaries preserve KYC records?"}'
pass() { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; exit 1; }
json() { python3 -c "import sys,json; d=json.load(sys.stdin); print($1)"; }

echo "Waiting for $BASE/readyz ..."
for _ in $(seq 1 90); do
  if curl -fsS "$BASE/readyz" >/dev/null 2>&1; then break; fi; sleep 2
done
curl -fsS "$BASE/readyz" | grep -q '"ready"' && pass "readyz: index, cache and models ready" || fail "not ready: $(curl -s $BASE/readyz)"

R1=$(curl -fsS -X POST "$BASE/v1/query" -H 'content-type: application/json' -H 'X-Request-ID: smoke-1' -d "$Q")
[ "$(echo "$R1" | json 'd["request_id"]')" = "smoke-1" ] && pass "request id propagated" || fail "request id"
[ "$(echo "$R1" | json 'len(d["citations"])>0 and not d["abstained"]')" = "True" ] && pass "answer has citations: $(echo "$R1" | json 'd["text"][:90]')" || fail "no citations: $R1"
echo "    latency_ms=$(echo "$R1" | json 'd["latency_ms"]["total"]') confidence=$(echo "$R1" | json 'round(d["confidence"],3)') model=$(echo "$R1" | json 'd["model"]')"

R2=$(curl -fsS -X POST "$BASE/v1/query" -H 'content-type: application/json' -d "$Q")
[ "$(echo "$R2" | json 'd["cached"]')" = "True" ] && pass "second identical request served from Redis cache ($(echo "$R2" | json 'd["latency_ms"]["total"]') ms)" || fail "cache miss on repeat"

S=$(curl -fsS -N -X POST "$BASE/v1/query/stream" -H 'content-type: application/json' -d '{"query":"What is the deadline for pay-out of funds and securities under T+1?"}')
echo "$S" | head -1 | grep -q '^event: sources' && echo "$S" | grep -q '^event: token' && echo "$S" | grep -q '^event: done' \
  && pass "SSE stream: sources -> token* -> done ($(echo "$S" | grep -c '^event: token') token events)" || fail "bad SSE stream: $S"

A=$(curl -fsS -X POST "$BASE/v1/query" -H 'content-type: application/json' -d '{"query":"Who won the football world cup in 2018?"}')
[ "$(echo "$A" | json 'd["abstained"]')" = "True" ] && pass "off-topic question abstains ($(echo "$A" | json 'd["abstain_reason"]'))" || fail "should abstain: $A"

CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/query" -H 'content-type: application/json' -d '{"query":""}')
[ "$CODE" = "422" ] && pass "invalid request rejected (422)" || fail "validation returned $CODE"
echo "All smoke checks passed."

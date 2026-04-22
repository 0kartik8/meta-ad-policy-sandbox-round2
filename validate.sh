#!/usr/bin/env bash
set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  exit 1
fi

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; }
fail() { log "${RED}FAILED${NC} -- $1"; exit 1; }

printf "\n${BOLD}=== OpenEnv Validator ===${NC}\n"

log "Step 1/3: Pinging HF Space ($PING_URL/reset) ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30 || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live!"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200). Is your Space running?"
fi

log "Step 2/3: Running docker build ..."
if docker build "$REPO_DIR" > /dev/null 2>&1; then
  pass "Docker build succeeded"
else
  fail "Docker build failed"
fi

log "Step 3/3: Running openenv validate ..."
if cd "$REPO_DIR" && openenv validate > /dev/null 2>&1; then
  pass "openenv validate passed"
else
  fail "openenv validate failed. Check openenv.yaml"
fi

printf "\n${GREEN}${BOLD} All 3/3 checks passed! Ready to submit.${NC}\n"
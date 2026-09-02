#!/usr/bin/env bash
# Continuous search for ONE game against the local pool. Running one of these
# per game gives game-level parallelism: a slow game (Dominion, ~15 min/run)
# can no longer starve the cheap ones, which is what a single sequential loop
# was doing. All per-game loops share the same container pool and cache.
#
#   GAME=CantStop ./scripts/search_game.sh
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export PYTHONPATH=src
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

GAME="${GAME:?set GAME}"
POOL="${POOL:-10}"          # this game's slice size
FIRST_PORT="${FIRST_PORT:-3000}"
POOL_TOTAL="${POOL_TOTAL:-10}"  # full pool to keep alive
BUDGET="${BUDGET:-40}"          # per pass; small so PBIL state updates often
REPEATS="${REPEATS:-2}"        # observations per config -> optimise the mean, not a lucky draw

echo $$ > "results/search-${GAME}.pid"
LOG="results/search-${GAME}.log"

heal() {
  docker info >/dev/null 2>&1 || { open -a Docker
    for _ in $(seq 1 40); do docker info >/dev/null 2>&1 && break; sleep 5; done; }
  local up; up=$(docker ps --filter 'name=ttb-api-' -q 2>/dev/null | wc -l | tr -d ' ')
  # Only one loop should rebuild the whole pool; a lock avoids 4 racing starts.
  if [ "${up:-0}" -lt "$POOL_TOTAL" ]; then
    if mkdir results/.heal.lock 2>/dev/null; then
      trap 'rmdir results/.heal.lock 2>/dev/null' RETURN
      ./scripts/localapi.sh start "$POOL_TOTAL" >> "$LOG" 2>&1
      rmdir results/.heal.lock 2>/dev/null
    else
      sleep 30   # another loop is healing; wait it out
    fi
  fi
}

pass=0
while true; do
  pass=$((pass + 1)); heal
  if [ $((pass % 2)) -eq 1 ]; then OPT=pbil; else OPT=ea; fi
  echo "=== $GAME pass $pass ($OPT) $(date '+%F %T') ===" >> "$LOG"
  python3 -u -m ttbalance --backend local --local-pool "$POOL" --workers "$POOL" \
      --first-port "$FIRST_PORT" --timeout-ms "${TIMEOUT_MS:-600000}" \
      --game "$GAME" --budget "$BUDGET" search --optimizer "$OPT" \
      --repeats "$REPEATS" --generations 500 --iterations 500 >> "$LOG" 2>&1
  sleep 1
done

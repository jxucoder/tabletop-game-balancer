#!/usr/bin/env bash
# Full-pool racing confirmation. Cheap games first so entries land fast. Races
# the top cached fast candidates with growing repeat counts (successive
# halving) and keeps the best *mean*, replacing n=1 lucky draws with confirmed
# configurations. Single client over all 10 containers -> no contention.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
POOL="${POOL:-10}"
LOG=results/verify.log
: > "$LOG"
for G in CantStop ExplodingKittens Wonders7 Dominion; do
  echo "=== verify $G $(date '+%T') ===" | tee -a "$LOG"
  python3 -u -m ttbalance --backend local --local-pool "$POOL" --workers "$POOL" \
      verify --game "$G" --run-type fast --from-run-type fast \
      --top 6 --keep 3 --rounds 1 3 5 7 >> "$LOG" 2>&1
done
python3 -u -m ttbalance best >> "$LOG" 2>&1
echo "VERIFY DONE $(date '+%T')" >> "$LOG"

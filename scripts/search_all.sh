#!/usr/bin/env bash
# One search loop per game, each pinned to a DISJOINT slice of the container
# pool so they never contend. Slice sizes follow headroom x cheapness: the
# cheap, least-optimised games get the most containers; Dominion (slow, already
# ~987/1000) gets one. Total must equal the running pool size (10).
#   game            : first_port : n_containers : per-pass budget
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
# game : first_port : n_containers : per-pass budget : repeats-per-config
# game : first_port : n_containers : budget : repeats : timeout_ms
# timeout is set ABOVE each game's legitimate run time so it kills only configs
# that hang the simulator (games that never terminate), never a slow-but-finite
# run. Dominion legitimately takes ~15 min, so its guard is 25 min.
declare -a PLAN=(
  "ExplodingKittens:3000:4:80:3:480000"   # ~1-2min runs; 8min guard
  "CantStop:3004:3:80:3:480000"           # ~3.5min runs; 8min guard
  "Wonders7:3007:2:60:2:480000"           # 8min guard
  "Dominion:3009:1:12:2:1500000"          # ~15min runs; 25min guard
)
for row in "${PLAN[@]}"; do
  IFS=: read -r g port n b reps tmo <<< "$row"
  GAME="$g" FIRST_PORT="$port" POOL="$n" BUDGET="$b" REPEATS="$reps" TIMEOUT_MS="$tmo" \
    nohup ./scripts/search_game.sh > "results/nohup-$g.out" 2>&1 &
  sleep 2
done
echo "launched ${#PLAN[@]} per-game loops on disjoint container slices"

#!/usr/bin/env bash
# Medium-fidelity search on Modal. fast is anti-correlated with medium for EK
# (rho=-0.79), so we search the trustworthy fidelity directly. Nori surrogate
# because medium evals are expensive (~15-35 min): the surrogate trains ONLY on
# medium observations for this game (feeding it fast would mislead it) and picks
# are evaluated at medium.
#   GAME=ExplodingKittens ./scripts/search_modal_medium.sh
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
source .venv-nori/bin/activate
export PYTHONPATH=src PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
GAME="${GAME:?}"; WORKERS="${WORKERS:-16}"; GENS="${GENS:-8}"
RUN_TYPE="${RUN_TYPE:-medium}"; TMO="${TMO:-3000000}"
echo $$ > "results/modal-med-$GAME.pid"; LOG="results/modal-med-$GAME.log"
pass=0
while true; do
  pass=$((pass+1))
  echo "=== modal-medium $GAME pass $pass $(date '+%T') ===" >> "$LOG"
  python -u -m ttbalance --backend modal --workers "$WORKERS" --run-type "$RUN_TYPE" \
    --timeout-ms "$TMO" \
    search --game "$GAME" --optimizer nori --generations "$GENS" \
    --opt-args '{"pool":2000,"batch":10,"init":22,"kappa":0.9,"pick_repeats":1}' \
    >> "$LOG" 2>&1
  python -u -m ttbalance best >> "$LOG" 2>&1
  sleep 2
done

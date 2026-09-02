#!/usr/bin/env bash
# Manage the pool of local evaluation API containers.
#
# One container runs exactly one simulation at a time — the server inside
# serialises requests — so parallelism comes from running several containers
# on consecutive ports and keeping one request in flight per container.
#
#   ./scripts/localapi.sh start [N]   # default 8
#   ./scripts/localapi.sh status
#   ./scripts/localapi.sh stop
set -uo pipefail
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
IMAGE=longhousedev/localapi
FIRST_PORT="${FIRST_PORT:-3000}"

cmd="${1:-status}"
n="${2:-12}"

case "$cmd" in
  start)
    docker info >/dev/null 2>&1 || { open -a Docker
      for _ in $(seq 1 40); do docker info >/dev/null 2>&1 && break; sleep 5; done; }
    docker pull -q "$IMAGE" >/dev/null 2>&1
    for i in $(seq 0 $((n-1))); do
      name="ttb-api-$i"
      docker rm -f "$name" >/dev/null 2>&1
      docker run -d --name "$name" --restart unless-stopped \
        --memory 1600m --cpus 1.5 -p "$((FIRST_PORT+i)):3000" "$IMAGE" >/dev/null
    done
    sleep 10
    echo "started $(docker ps --filter 'name=ttb-api-' -q | wc -l | tr -d ' ') containers"
    ;;
  stop)
    docker ps -a --filter "name=ttb-api-" -q | xargs -r docker rm -f >/dev/null
    echo "stopped"
    ;;
  status)
    docker ps --filter "name=ttb-api-" --format "{{.Names}}\t{{.Status}}\t{{.Ports}}" | sort -V
    running=0
    for c in $(docker ps --filter "name=ttb-api-" --format "{{.Names}}"); do
      k=$(docker exec "$c" sh -c 'ps -eo args | grep -c "[T]AG.jar"' 2>/dev/null | tr -dc '0-9')
      running=$((running + ${k:-0}))
    done
    echo "simulations in flight: $running"
    ;;
  *) echo "usage: $0 {start [N]|stop|status}"; exit 1 ;;
esac

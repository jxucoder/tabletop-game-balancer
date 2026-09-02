#!/usr/bin/env bash
# Bootstrap a fleet of TAG evaluation API containers on a fresh cloud VM.
# Run this ON THE VM (Ubuntu/Debian, root or sudo). One container pins ~1 core,
# so N should be roughly (vCPUs - 2). Ports are FIRST_PORT .. FIRST_PORT+N-1.
#
#   curl -fsSL <this> | sudo N=90 bash        # or copy the file over and run it
#   sudo N=90 FIRST_PORT=3000 ./cloud_localapi.sh
#
# Afterwards, from your laptop point the toolkit at it:
#   TTB_POOL_HOSTS="<VM_PUBLIC_IP>:3000:90" \
#     python -m ttbalance --backend local search --game Wonders7 --optimizer pbil ...
# (Open the port range in the VM firewall/security group first, ideally scoped
#  to your own IP — the API has no auth.)
set -euo pipefail
IMAGE="${IMAGE:-longhousedev/localapi}"
N="${N:-90}"
FIRST_PORT="${FIRST_PORT:-3000}"
MEM="${MEM:-1600m}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[cloud] installing docker..."
  curl -fsSL https://get.docker.com | sh
fi
echo "[cloud] pulling $IMAGE..."
docker pull -q "$IMAGE" >/dev/null

echo "[cloud] removing any old ttb-api containers..."
docker ps -aq --filter "name=ttb-api-" | xargs -r docker rm -f >/dev/null

echo "[cloud] starting $N containers on ports $FIRST_PORT..$((FIRST_PORT+N-1))..."
for i in $(seq 0 $((N-1))); do
  docker run -d --name "ttb-api-$i" --restart unless-stopped \
    --memory "$MEM" --cpus 1.0 -p "$((FIRST_PORT+i)):3000" "$IMAGE" >/dev/null
done
sleep 12
up=$(docker ps --filter "name=ttb-api-" -q | wc -l | tr -d ' ')
ip=$(curl -fsS ifconfig.me 2>/dev/null || echo "<VM_PUBLIC_IP>")
echo "[cloud] $up/$N containers up."
echo "[cloud] point the toolkit at:  TTB_POOL_HOSTS=\"$ip:$FIRST_PORT:$N\""
echo "[cloud] stop all:  docker ps -aq --filter name=ttb-api- | xargs -r docker rm -f"

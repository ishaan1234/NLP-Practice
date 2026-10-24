#!/bin/bash
# Start (or restart) the Room Twin GPU worker on a machine set up with setup.sh, and expose it
# through a free Cloudflare quick tunnel. Prints the two values the app needs:
#   ROOM_TWIN_GPU_URL=https://....trycloudflare.com
#   ROOM_TWIN_GPU_TOKEN=...
# The token is kept in /content/.worker_token so restarts keep it (the tunnel URL changes).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=${LOG:-/content/worker-logs}
mkdir -p "$LOG" /content/bin
[ -s /content/.worker_token ] || python3 -c "import secrets;print(secrets.token_urlsafe(32))" > /content/.worker_token
chmod 600 /content/.worker_token
export ROOM_TWIN_GPU_TOKEN=$(cat /content/.worker_token)
export SAM3D_REPO=${SAM3D_REPO:-/content/sam-3d-objects}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/content/mamba}
export HF_TOKEN=${HF_TOKEN:-$(cat /content/.hf_token 2>/dev/null)}

# stop a previous run (pid files, so nothing else gets matched by name)
for f in "$LOG"/*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null || true; rm -f "$f"; done
sleep 1

# SAM 3D Objects in its own environment (port 8002, local only)
nohup /content/bin/micromamba run -n sam3d-objects \
  python "$HERE/worker.py" sam3d --port 8002 > "$LOG/sam3d.log" 2>&1 &
echo $! > "$LOG/sam3d.pid"
# SAM 3 + the public endpoint (port 8000, local only; the tunnel exposes it)
nohup python3 "$HERE/worker.py" main --port 8000 > "$LOG/main.log" 2>&1 &
echo $! > "$LOG/main.pid"

# Cloudflare quick tunnel: HTTPS URL, no account needed
if [ ! -x /content/bin/cloudflared ]; then
  curl -Ls -o /content/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /content/bin/cloudflared
fi
: > "$LOG/tunnel.log"
nohup /content/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 > "$LOG/tunnel.log" 2>&1 &
echo $! > "$LOG/tunnel.pid"
URL=""
for _ in $(seq 1 60); do
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG/tunnel.log" | head -1)
  [ -n "$URL" ] && break
  sleep 1
done
# wait until the worker answers (first start loads nothing yet: models load on first request)
for _ in $(seq 1 60); do
  curl -s -o /dev/null -H "X-Worker-Token: $ROOM_TWIN_GPU_TOKEN" http://127.0.0.1:8000/health && break
  sleep 1
done
echo "ROOM_TWIN_GPU_URL=$URL"
echo "ROOM_TWIN_GPU_TOKEN=$ROOM_TWIN_GPU_TOKEN"

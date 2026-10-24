#!/bin/bash
# One command (Linux / macOS / WSL): a Colab GPU running the Room Twin GPU worker.
#   pip install google-colab-cli
#   export HF_TOKEN=hf_...      # access to facebook/sam3 + facebook/sam-3d-objects
#   ./gpu-worker/colab.sh       # prints ROOM_TWIN_GPU_URL / ROOM_TWIN_GPU_TOKEN
# First run installs for ~20-40 min; later runs on the same session reuse the install.
# Uses Colab compute units while the session runs; stop it with: colab stop -s roomtwin
set -e
cd "$(dirname "$0")/.."
SESSION=${SESSION:-roomtwin}
GPU=${GPU:-A100}
colab() { command colab --auth "${COLAB_AUTH:-oauth2}" "$@"; }
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face token with access to facebook/sam3 and facebook/sam-3d-objects}"

if ! colab sessions 2>/dev/null | grep -qw "$SESSION"; then
  echo "[roomtwin] starting a Colab $GPU session '$SESSION'..." >&2
  colab new -s "$SESSION" --gpu "$GPU" >&2
fi

tmp=$(mktemp -d)
tar -czf "$tmp/rt.tgz" backend/app gpu-worker
printf %s "$HF_TOKEN" > "$tmp/tok" && chmod 600 "$tmp/tok"
colab upload -s "$SESSION" "$tmp/rt.tgz" /content/rt.tgz >&2
colab upload -s "$SESSION" "$tmp/tok" /content/.hf_token >&2
rm -rf "$tmp"

# unpack + (re)run setup in the background on the VM; it skips what is already installed
colab exec -s "$SESSION" >&2 <<'PY'
import os, subprocess
subprocess.run("mkdir -p /content/room-twin && tar -xzf /content/rt.tgz -C /content/room-twin && chmod 600 /content/.hf_token",
               shell=True, check=True)
if not os.path.exists("/content/.setup_done"):
    subprocess.Popen("bash /content/room-twin/gpu-worker/setup.sh > /content/setup.log 2>&1 "
                     "&& grep -q SETUP_DONE /content/setup.log && touch /content/.setup_done", shell=True)
print("[roomtwin] setup", "already done" if os.path.exists("/content/.setup_done") else "running (first time: 20-40 min)")
PY

# wait for setup, polling in short calls
while true; do
  state=$(colab exec -s "$SESSION" 2>/dev/null <<'PY'
import os, subprocess
if os.path.exists("/content/.setup_done") or "SETUP_DONE" in open("/content/setup.log", errors="ignore").read()[-2000:]:
    print("DONE")
elif subprocess.run("pgrep -f 'gpu-worker/[s]etup.sh'", shell=True, capture_output=True).returncode:
    print("FAILED")
else:
    last = open("/content/setup.log", errors="ignore").read()[-4000:].replace("\r", "\n").strip().splitlines()
    print("RUNNING " + (last[-1].strip()[:100] if last else ""))  # progress bars redraw with \r
PY
)
  case "$state" in
    *DONE*) break ;;
    *FAILED*) echo "[roomtwin] setup failed; see: colab exec -s $SESSION <<< 'print(open(\"/content/setup.log\").read()[-3000:])'" >&2; exit 1 ;;
    *) echo "[roomtwin] installing... ${state#*RUNNING }" >&2; sleep 30 ;;
  esac
done

colab exec -s "$SESSION" <<'PY'
import subprocess
print(subprocess.run(["bash", "/content/room-twin/gpu-worker/start.sh"], capture_output=True, text=True).stdout)
PY

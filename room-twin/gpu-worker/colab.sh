#!/bin/bash
# One command from your laptop: a Colab A100 running the Room Twin GPU worker.
#   pip install google-colab-cli && export HF_TOKEN=hf_...   # access to facebook/sam3 + sam-3d-objects
#   ./gpu-worker/colab.sh                                    # prints ROOM_TWIN_GPU_URL / _TOKEN
# Uses Colab compute units while the session runs; stop it with: colab stop -s roomtwin
set -e
cd "$(dirname "$0")/.."
SESSION=${SESSION:-roomtwin}
GPU=${GPU:-A100}
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face token with access to facebook/sam3 and facebook/sam-3d-objects}"
colab sessions 2>/dev/null | grep -q "$SESSION" || colab new -s "$SESSION" --gpu "$GPU"
tmp=$(mktemp -d)
tar -czf "$tmp/rt.tgz" backend/app gpu-worker
printf %s "$HF_TOKEN" > "$tmp/tok" && chmod 600 "$tmp/tok"
colab upload -s "$SESSION" "$tmp/rt.tgz" /content/rt.tgz
colab upload -s "$SESSION" "$tmp/tok" /content/.hf_token
rm -rf "$tmp"
colab exec -s "$SESSION" <<'PY'
import subprocess
print(subprocess.run(["bash", "-c", "mkdir -p /content/room-twin && tar -xzf /content/rt.tgz -C /content/room-twin "
                      "&& chmod 600 /content/.hf_token && bash /content/room-twin/gpu-worker/setup.sh > /content/setup.log 2>&1; "
                      "tail -3 /content/setup.log"], capture_output=True, text=True).stdout)
PY
colab exec -s "$SESSION" <<'PY'
import subprocess
print(subprocess.run(["bash", "/content/room-twin/gpu-worker/start.sh"], capture_output=True, text=True).stdout)
PY

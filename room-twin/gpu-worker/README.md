# Room Twin GPU worker

Real 3D models of your furniture need **SAM 3D Objects**, which needs a Linux machine with an
**NVIDIA GPU with at least 32 GB of memory** (no Mac, almost no laptop). This worker runs SAM 3
(segmentation) and SAM 3D Objects (photo → 3D mesh) on such a machine and serves them over
HTTPS, so the app on your laptop or on Vercel uses them remotely.

## On Google Colab (A100, uses compute units)

```bash
pip install google-colab-cli
export HF_TOKEN=hf_...      # Hugging Face token with access to facebook/sam3 and facebook/sam-3d-objects
./gpu-worker/colab.sh       # from room-twin/: starts an A100, installs (~20-30 min first time), prints:
#   ROOM_TWIN_GPU_URL=https://<random>.trycloudflare.com
#   ROOM_TWIN_GPU_TOKEN=<secret>
```

The first `colab` command asks you to sign in with Google (open the printed link, paste the
code back). Request access to both models on huggingface.co first - they are gated.

Then point the app at it:

- **Locally:** `export ROOM_TWIN_GPU_URL=... ROOM_TWIN_GPU_TOKEN=...` before `uvicorn ...`
- **Vercel:** add both as Environment Variables in the project settings and redeploy.

A build then shows the room straight away with photo cut-outs, and the editor turns each
detected object into a real 3D model one at a time (each request stays inside Vercel's time
limit). Any cut-out also has a **Make 3D model** button.

Stop the session when you're done (`colab stop -s roomtwin`): an A100 uses compute units while
it runs. The tunnel URL changes every time the worker restarts.

## On any other GPU machine

```bash
HF_TOKEN=hf_... bash gpu-worker/setup.sh     # paths default to /content; edit for your machine
bash gpu-worker/start.sh
```

## Pieces

| File | What it does |
|---|---|
| `worker.py` | FastAPI: `/segment` (SAM 3) and `/reconstruct` (SAM 3D), token-protected |
| `setup.sh` | micromamba env for SAM 3D Objects (Python 3.11 / CUDA 12.1), SAM 3, both checkpoints |
| `start.sh` | runs the two worker processes + a Cloudflare quick tunnel, prints URL and token |
| `colab.sh` | does all of the above on a Colab GPU from your laptop |

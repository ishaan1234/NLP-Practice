#!/usr/bin/env python3
"""Room Twin launcher - one command on Windows, macOS or Linux.

    python launch.py           set up, start a Colab GPU running SAM 3 + SAM 3D (real 3D models),
                               start the app connected to it, open the browser
    python launch.py --no-gpu  just the app, no Colab (photo cut-outs, CPU segmentation)
    python launch.py stop      stop the Colab GPU session (stops using compute units)
    python launch.py status    show what is configured

Needs Hugging Face access to facebook/sam3 (and facebook/sam-3d-objects; until then TRELLIS is used), and on Windows WSL
(`wsl --install` once), because Google's Colab CLI runs on Linux/macOS only.

Settings live in .roomtwin.env (not committed).
Only the Python standard library is used here.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".roomtwin.env"
VENV = ROOT / ".venv"
WINDOWS = platform.system() == "Windows"
SESSION = os.environ.get("ROOMTWIN_COLAB_SESSION", "roomtwin")
PORT = int(os.environ.get("ROOMTWIN_PORT", "8000"))


def say(msg: str):
    print(f"[roomtwin] {msg}", flush=True)


# ---------------------------------------------------------------- settings
def load_env() -> dict:
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def save_env(env: dict):
    ENV_FILE.write_text("# Room Twin local settings (keep private, not committed)\n"
                        + "".join(f"{k}={v}\n" for k, v in env.items() if v))
    try:
        os.chmod(ENV_FILE, 0o600)
    except OSError:
        pass


def ask_secret(env: dict, key: str, prompt: str, required: bool) -> str:
    if os.environ.get(key):
        env[key] = os.environ[key]
    if not env.get(key):
        if not sys.stdin.isatty():  # no one to ask (started from a script)
            if required:
                sys.exit(f"{key} is needed: set it in {ENV_FILE.name} or the environment.")
            return ""
        try:
            value = getpass.getpass(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            value = ""
        if not value and required:
            sys.exit(f"{key} is needed for this.")
        env[key] = value
        save_env(env)
    return env.get(key, "")


# ---------------------------------------------------------------- python environment for the app
def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")


def ensure_venv():
    req = ROOT / "backend" / "requirements.txt"
    stamp = VENV / ".requirements.sha"
    digest = hashlib.sha256(req.read_bytes()).hexdigest()
    if not venv_python().exists():
        say("creating a Python environment for the app (.venv)...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    if not stamp.exists() or stamp.read_text() != digest:
        say("installing the app's packages (first run takes a minute)...")
        subprocess.run([str(venv_python()), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
        subprocess.run([str(venv_python()), "-m", "pip", "install", "-q", "-r", str(req)], check=True)
        stamp.write_text(digest)


# ---------------------------------------------------------------- Colab CLI (natively or through WSL)
def shell(cmd: str, env: dict | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a bash command where the Colab CLI lives: here on macOS/Linux, inside WSL on Windows."""
    cmd = 'export PATH="$HOME/.local/bin:$PATH"; ' + cmd
    full_env = {**os.environ, **(env or {})}
    if WINDOWS:
        if not shutil.which("wsl"):
            sys.exit("The Colab CLI needs WSL on Windows. Install it once (admin PowerShell): wsl --install\n"
                     "Or start without a GPU: python launch.py --no-gpu")
        # pass selected variables into WSL
        full_env["WSLENV"] = ":".join([*(f"{k}/u" for k in (env or {})), full_env.get("WSLENV", "")]).strip(":")
        args = ["wsl", "bash", "-lc", cmd]
    else:
        args = ["bash", "-lc", cmd]
    return subprocess.run(args, env=full_env, text=True, stdout=subprocess.PIPE if capture else None)


def repo_path_for_shell() -> str:
    if not WINDOWS:
        return str(ROOT)
    out = subprocess.run(["wsl", "wslpath", "-a", str(ROOT)], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def ensure_colab_cli():
    if shell("command -v colab >/dev/null", capture=True).returncode == 0:
        return
    say("installing the Colab CLI (via uv, no admin rights needed)...")
    r = shell("command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; "
              '"$HOME/.local/bin/uv" tool install google-colab-cli')
    if r.returncode:
        sys.exit("Couldn't install the Colab CLI. Try manually: pip install google-colab-cli")


def colab_login():
    """First use: the CLI prints a Google sign-in link and asks for the code - right here."""
    say("checking the Colab sign-in (first time: open the link it prints and paste the code)...")
    if shell("colab --auth oauth2 sessions").returncode:
        sys.exit("Colab sign-in failed.")


def worker_alive(url: str, token: str) -> bool:
    if not url or not token:
        return False
    try:
        req = urllib.request.Request(url.rstrip("/") + "/health", headers={"X-Worker-Token": token})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status == 200 and json.loads(r.read()).get("role") == "main"
    except Exception:
        return False


def hf_access(token: str) -> list[str]:
    """Gated models this token can't download yet (checked in seconds, before a 30-min install)."""
    blocked = []
    for repo, f in (("facebook/sam-3d-objects", ".gitattributes"), ("facebook/sam3", "config.json")):
        req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{f}", method="HEAD",
                                     headers={"Authorization": f"Bearer {token}"})
        try:
            urllib.request.urlopen(req, timeout=20)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                blocked.append(repo)
        except Exception:
            pass  # offline etc.: let the real install report it
    return blocked


def start_gpu(env: dict) -> bool:
    if worker_alive(env.get("ROOM_TWIN_GPU_URL", ""), env.get("ROOM_TWIN_GPU_TOKEN", "")):
        say(f"GPU worker already running at {env['ROOM_TWIN_GPU_URL']}")
        return True
    hf = ask_secret(env, "HF_TOKEN", "Hugging Face token (access to facebook/sam3 + facebook/sam-3d-objects): ",
                    required=True)
    blocked = hf_access(hf)
    if blocked == ["facebook/sam-3d-objects"]:
        say("no access to facebook/sam-3d-objects yet: using TRELLIS for the 3D models meanwhile")
        say("  (the worker switches to SAM 3D by itself once access is granted and it's set up again)")
        blocked = []
    if blocked:
        say("your Hugging Face token can't download: " + ", ".join(blocked))
        say("  1. on huggingface.co open each model page and request access (Meta approves by hand)")
        say("  2. use a classic 'Read' token, or tick 'Read access to contents of all public gated repos")
        say("     you can access' on a fine-grained one; then edit HF_TOKEN in .roomtwin.env")
        say("starting without the GPU for now (photo cut-outs)")
        return False
    ensure_colab_cli()
    colab_login()
    say("starting the GPU worker on Colab (first time: installs for 20-40 min; it says what it's doing)...")
    r = shell(f'cd "{repo_path_for_shell()}" && COLAB_AUTH=oauth2 SESSION={SESSION} bash gpu-worker/colab.sh',
              env={"HF_TOKEN": hf}, capture=True)
    for line in (r.stdout or "").splitlines():
        if line.startswith(("ROOM_TWIN_GPU_URL=", "ROOM_TWIN_GPU_TOKEN=")):
            k, v = line.split("=", 1)
            env[k] = v.strip()
    if r.returncode or not env.get("ROOM_TWIN_GPU_URL"):
        say("the GPU worker didn't start; continuing without it (photo cut-outs only)")
        return False
    save_env(env)
    if not worker_alive(env["ROOM_TWIN_GPU_URL"], env["ROOM_TWIN_GPU_TOKEN"]):
        say("the GPU worker started but doesn't answer yet; it may need a minute")
    say(f"GPU worker: {env['ROOM_TWIN_GPU_URL']}")
    return True


# ---------------------------------------------------------------- commands
def up(colab: bool):
    env = load_env()
    ensure_venv()
    ask_secret(env, "ANTHROPIC_API_KEY", "Anthropic API key for the AI assistant (Enter to skip): ", required=False)
    gpu = colab and start_gpu(env)
    app_env = {**os.environ, **{k: v for k, v in env.items() if v}}
    if not gpu:
        app_env.pop("ROOM_TWIN_GPU_URL", None)
        app_env.pop("ROOM_TWIN_GPU_TOKEN", None)
    url = f"http://localhost:{PORT}"
    say(f"starting Room Twin at {url}  (phone capture: same Wi-Fi; Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        subprocess.run([str(venv_python()), "-m", "uvicorn", "app.main:app", "--app-dir", "backend",
                        "--host", "0.0.0.0", "--port", str(PORT)], cwd=ROOT, env=app_env)
    except KeyboardInterrupt:
        pass
    if gpu:
        say("app stopped. The Colab GPU keeps running (and using compute units) until: python launch.py stop")


def stop():
    ensure_colab_cli()
    shell(f"colab --auth oauth2 stop -s {SESSION}")
    env = load_env()
    env.pop("ROOM_TWIN_GPU_URL", None)
    save_env(env)
    say("Colab GPU session stopped.")


def status():
    env = load_env()
    say(f"settings file: {ENV_FILE} ({'found' if ENV_FILE.exists() else 'not yet'})")
    for key in ("HF_TOKEN", "ANTHROPIC_API_KEY", "ROOM_TWIN_GPU_TOKEN"):
        say(f"{key}: {'set' if env.get(key) else 'not set'}")
    url = env.get("ROOM_TWIN_GPU_URL", "")
    say(f"GPU worker: {url or 'none'} - {'answering' if worker_alive(url, env.get('ROOM_TWIN_GPU_TOKEN', '')) else 'not answering'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "stop" in args:
        stop()
    elif "status" in args:
        status()
    else:
        up(colab="--no-gpu" not in args)

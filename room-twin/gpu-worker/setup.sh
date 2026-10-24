#!/bin/bash
# Room Twin GPU worker setup on a Linux NVIDIA machine (e.g. a Colab A100; SAM 3D Objects needs
# >= 32 GB of GPU memory): SAM 3D Objects in its own micromamba environment, SAM 3 in the
# machine's Python, and both gated checkpoints. Needs a Hugging Face token with access to
# facebook/sam3 and facebook/sam-3d-objects in /content/.hf_token (or $HF_TOKEN).
set -x
cd /content
export HF_TOKEN=${HF_TOKEN:-$(cat /content/.hf_token 2>/dev/null)}
mkdir -p ~/.cache/huggingface && [ -n "$HF_TOKEN" ] && printf %s "$HF_TOKEN" > ~/.cache/huggingface/token
# --- micromamba
if [ ! -x /content/bin/micromamba ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /content bin/micromamba
fi
export MAMBA_ROOT_PREFIX=/content/mamba
eval "$(/content/bin/micromamba shell hook -s bash)"
# --- SAM 3D Objects
[ -d sam-3d-objects ] || git clone --depth 1 https://github.com/facebookresearch/sam-3d-objects
cd sam-3d-objects
micromamba env list | grep -q sam3d-objects || micromamba create -y -f environments/default.yml
micromamba activate sam3d-objects
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"  # pypi.ngc.nvidia.com is unreachable from Colab
pip install -e '.[dev]'
pip install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
./patching/hydra
pip install 'huggingface-hub[cli]<1.0' fastapi uvicorn python-multipart
python -c "import torch;print('SAM3D ENV torch', torch.__version__, torch.cuda.is_available())"
# gated checkpoints
if [ ! -f checkpoints/hf/pipeline.yaml ]; then
  hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects
  mv checkpoints/hf-download/checkpoints checkpoints/hf && rm -rf checkpoints/hf-download
fi
micromamba deactivate
cd /content
# --- SAM 3 (Colab's own Python)
[ -d sam3 ] || git clone --depth 1 https://github.com/facebookresearch/sam3
pip install -e sam3
python -c "import sam3;print('SAM3 import ok')"
pip install fastapi uvicorn python-multipart requests
echo SETUP_DONE

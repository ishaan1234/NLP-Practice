#!/bin/bash
# Stand-in 3D model while SAM 3D Objects access is pending: Microsoft TRELLIS (MIT, ungated
# weights microsoft/TRELLIS-image-large), installed into the sam3d-objects environment, which
# already has torch 2.5.1/cu121, spconv, kaolin and flash-attn.
set -x
export MAMBA_ROOT_PREFIX=/content/mamba
eval "$(/content/bin/micromamba shell hook -s bash)"
micromamba activate sam3d-objects
cd /content
[ -d TRELLIS ] || git clone --recurse-submodules --depth 1 https://github.com/microsoft/TRELLIS
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja \
  trimesh xatlas pyvista pymeshfix igraph transformers
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
export TORCH_CUDA_ARCH_LIST="8.0"   # A100
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
pip install --no-build-isolation git+https://github.com/JeffreyXiang/diffoctreerast.git
[ -d /tmp/mip-splatting ] || git clone --depth 1 https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting
pip install --no-build-isolation /tmp/mip-splatting/submodules/diff-gaussian-rasterization/
pip install "numpy<2" "opencv-python-headless<4.11"  # kaolin's binaries need numpy 1.x
python -c "import sys; sys.path.insert(0, '/content/TRELLIS'); import os; os.environ['SPCONV_ALGO']='native'; \
from trellis.pipelines import TrellisImageTo3DPipeline; print('TRELLIS import ok')" || exit 1
echo TRELLIS_DONE

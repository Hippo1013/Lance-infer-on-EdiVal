#!/usr/bin/env bash
# Server-only scoring environments. No edits to Lance or shared environments.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${SCORING_SETUP_OUTPUT:-$project_dir/outputs/setup/scoring_20261004}"
conda_exe="${CONDA_EXE:-/media/damoxing/tangzecong/miniconda3/bin/conda}"
env_root=/home/chs/conda/envs
model_root=/home/chs/model
mkdir -p "$output"
setup_tmp="$(mktemp -d /tmp/umm-scoring-install.XXXXXX)"
export TMPDIR="$setup_tmp" PIP_CACHE_DIR="$setup_tmp/pip-cache"
export PIP_INDEX_URL=https://pypi.org/simple
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4
# Compile without running a GPU kernel, so burn can remain active.
export CUDA_VISIBLE_DEVICES=""
cleanup() {
  rc=$?
  if [[ "$rc" == 0 ]]; then echo ready > "$output/environment.status";
  else echo "failed:$rc" > "$output/environment.status"; fi
  rm -r -- "$setup_tmp"
}
trap cleanup EXIT
echo creating > "$output/environment.status"
if [[ -f /home/chs/net_proxy/proxy-off.sh ]]; then
  source /home/chs/net_proxy/proxy-off.sh >/dev/null
fi
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
for name in mice-metrics mice-judge; do
  if [[ ! -x "$env_root/$name/bin/python" ]]; then
    "$conda_exe" create -y -p "$env_root/$name" --override-channels -c conda-forge python=3.12 pip
  fi
done
"$conda_exe" install -y -p "$env_root/mice-metrics" --override-channels -c conda-forge libxcb libgl libglib
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:$env_root/mice-metrics/lib:${LD_LIBRARY_PATH:-}"
metrics="$env_root/mice-metrics/bin/python"
judge="$env_root/mice-judge/bin/python"
echo installing_judge > "$output/environment.status"
"$judge" -m pip install 'vllm==0.30.0' 'transformers==5.14.1'
"$judge" -m pip check
echo installing_metrics > "$output/environment.status"
"$metrics" -m pip install 'torch==2.13.0' 'torchvision==0.28.0' \
  'transformers==4.53.3' 'timm==1.0.25' 'Flask==3.1.3' \
  'requests==2.32.5' 'supervision==0.27.0.post1' 'addict==2.4.0' \
  'yapf==0.43.0' pycocotools ninja setuptools wheel
echo building_groundingdino > "$output/environment.status"
"$metrics" - "$project_dir" "$setup_tmp" "$model_root" <<'PY'
import hashlib, io, json, pathlib, tarfile, sys
import requests
project, tmp, models = map(pathlib.Path, sys.argv[1:])
revision = json.loads((project / 'configs/scoring_models.json').read_text())['groundingdino']['source_revision']
url = f'https://codeload.github.com/IDEA-Research/GroundingDINO/tar.gz/{revision}'
r = requests.get(url, timeout=120)
r.raise_for_status()
with tarfile.open(fileobj=io.BytesIO(r.content)) as tar:
    tar.extractall(tmp, filter='data')
src = tmp / f'GroundingDINO-{revision}'
# Tensor API compatibility only: no numerical algorithm or thresholds changed.
for rel in ['ms_deform_attn_cuda.cu', 'ms_deform_attn.h']:
    p = src / 'groundingdino/models/GroundingDINO/csrc/MsDeformAttn' / rel
    text = p.read_text()
    text = text.replace('.type().is_cuda()', '.is_cuda()')
    text = text.replace('AT_DISPATCH_FLOATING_TYPES(value.type(),', 'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),')
    text = text.replace('.data<', '.data_ptr<')
    p.write_text(text)
dest = models / 'GroundingDINO-SwinT-OGC'
dest.mkdir(parents=True, exist_ok=True)
config = (src / 'groundingdino/config/GroundingDINO_SwinT_OGC.py').read_text()
(dest / 'GroundingDINO_SwinT_OGC.py').write_text(config)
local_config = config.replace('text_encoder_type = "bert-base-uncased"',
                              f'text_encoder_type = "{models / "bert-base-uncased"}"')
(dest / 'GroundingDINO_SwinT_OGC.local.py').write_text(local_config)
(dest / 'source.json').write_text(json.dumps({'revision': revision, 'archive_sha256': hashlib.sha256(r.content).hexdigest(),
    'patch': 'Tensor.type API migration: is_cuda/scalar_type/data_ptr; no numerical changes'}, indent=2) + '\n')
PY
revision="$("$metrics" -c "import json; print(json.load(open('$project_dir/configs/scoring_models.json'))['groundingdino']['source_revision'])")"
# opencv-python is required by upstream package metadata; install it once here.
"$metrics" -m pip install opencv-python
"$metrics" -m pip install --no-build-isolation --no-deps "$setup_tmp/GroundingDINO-$revision"
"$metrics" -m pip check
"$metrics" -c 'import torch; from groundingdino import _C; import timm; assert "vit_large_patch16_dinov3" in timm.list_models(); print("Native detection extension and DINOv3 architecture available")'
"$judge" -c 'from transformers import AutoConfig; from vllm.model_executor.models import ModelRegistry; print("vLLM import passed")'
"$metrics" -m pip freeze > "$output/metrics.freeze.txt"
"$judge" -m pip freeze > "$output/judge.freeze.txt"
"$conda_exe" list -p "$env_root/mice-metrics" --explicit > "$output/metrics.conda-explicit.txt"
"$conda_exe" list -p "$env_root/mice-judge" --explicit > "$output/judge.conda-explicit.txt"
echo 'Environment installation complete; GPU smoke validation remains separate.'

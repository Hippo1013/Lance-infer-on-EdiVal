#!/usr/bin/env bash
# Independent official EdiVal component environments; no edits to existing envs.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${EDIVAL_SETUP_OUTPUT:-$project_dir/outputs/setup/edival_official_20261006}"
conda_exe="${CONDA_EXE:-/media/damoxing/tangzecong/miniconda3/bin/conda}"
env_root=/home/chs/conda/envs
mkdir -p "$output"
setup_tmp="$(mktemp -d /tmp/umm-edival-install.XXXXXX)"
export TMPDIR="$setup_tmp" PIP_CACHE_DIR="$setup_tmp/pip-cache"
export PIP_INDEX_URL=https://pypi.org/simple
export CUDA_VISIBLE_DEVICES="" DS_BUILD_OPS=0
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}"
cleanup() {
  rc=$?
  if [[ "$rc" == 0 ]]; then echo installed > "$output/environment.status";
  else echo "failed:$rc" > "$output/environment.status"; fi
  rm -r -- "$setup_tmp"
}
trap cleanup EXIT
source /home/chs/net_proxy/proxy-off.sh >/dev/null
export http_proxy=http://192.168.32.28:18000 https_proxy=http://192.168.32.28:18000
echo cloning > "$output/environment.status"
for entry in EdiVal:mice-metrics EdiVal-judge:mice-judge EdiVal-hps:mice-metrics; do
  name="${entry%%:*}"; parent="${entry##*:}"
  if [[ ! -e "$env_root/$name" ]]; then
    "$conda_exe" create -y -p "$env_root/$name" --clone "$env_root/$parent" --offline
  else
    echo "Refusing to alter an existing environment without checking: $env_root/$name" >&2
    exit 1
  fi
done
"$conda_exe" install -y -p "$env_root/EdiVal-judge" --offline --override-channels -c conda-forge libxcb libgl libglib
printf '%s\n' 'torch==2.13.0' 'torchvision==0.28.0' > "$setup_tmp/torch-constraints.txt"
echo metrics_dependencies > "$output/environment.status"
"$env_root/EdiVal/bin/python" -m pip install -c "$setup_tmp/torch-constraints.txt" \
  'transformers==4.57.6' 'accelerate==1.12.0' gdown pandas tabulate sentencepiece
echo judge_dependencies > "$output/environment.status"
"$env_root/EdiVal-judge/bin/python" -m pip install -c "$setup_tmp/torch-constraints.txt" \
  'timm==1.0.25' 'supervision==0.27.0.post1' 'addict==2.4.0' 'yapf==0.43.0' \
  gdown pandas tabulate matplotlib pycocotools opencv-python
# Reuse the already validated native GroundingDINO extension: same Python/Torch ABI.
"$env_root/EdiVal-judge/bin/python" - "$env_root" <<'PY'
import pathlib, shutil, sys
root = pathlib.Path(sys.argv[1])
src = root / 'mice-metrics/lib/python3.12/site-packages'
dst = root / 'EdiVal-judge/lib/python3.12/site-packages'
for name in ['groundingdino', 'groundingdino-0.1.0.dist-info']:
    shutil.copytree(src / name, dst / name)
PY
echo hps_dependencies > "$output/environment.status"
"$env_root/EdiVal-hps/bin/python" -m pip install -c "$setup_tmp/torch-constraints.txt" \
  'hpsv3==1.0.0' 'transformers==4.45.2' 'accelerate==0.34.2' \
  'peft==0.13.2' 'trl==0.11.4' 'datasets==3.0.2' 'diffusers==0.30.3' \
  matplotlib pandas tensorboard tabulate rich
echo importing > "$output/environment.status"
for name in EdiVal EdiVal-judge EdiVal-hps; do
  export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:$env_root/$name/lib"
  "$env_root/$name/bin/python" -m pip check > "$output/$name.pip-check.txt"
  "$env_root/$name/bin/python" -m pip freeze > "$output/$name.freeze.txt"
  "$conda_exe" list -p "$env_root/$name" --explicit > "$output/$name.conda-explicit.txt"
done
"$env_root/EdiVal/bin/python" -c 'import torch; from groundingdino import _C; from transformers import DINOv3ViTModel, ViTModel, T5ForConditionalGeneration; import cv2; print("EdiVal imports passed")'
"$env_root/EdiVal-judge/bin/python" -c 'from vllm import LLM, SamplingParams; print("EdiVal-judge imports passed")'
"$env_root/EdiVal-hps/bin/python" -c 'from hpsv3 import HPSv3RewardInferencer; print("EdiVal-hps imports passed")'
echo 'Dependencies installed; model/GPU validation remains separate.'

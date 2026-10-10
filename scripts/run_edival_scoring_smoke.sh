#!/usr/bin/env bash
# GPU1 must already be released using the workspace burn procedure.
set -euo pipefail
gpu="${1:?GPU index required}"; shift
[[ "$gpu" == 1 ]] || { echo 'This EdiVal validation is authorized on physical GPU1 only' >&2; exit 2; }
output="${1:?Independent setup validation directory required}"; shift
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
(( used <= 64 )) || { echo "GPU $gpu occupied: $used MiB" >&2; exit 1; }
smoke_tmp="$(mktemp -d /tmp/umm-edival-smoke.XXXXXX)"
system_library_path="${LD_LIBRARY_PATH:-}"
export TMPDIR="$smoke_tmp" XDG_CACHE_HOME="$smoke_tmp/cache"
export PYTHONDONTWRITEBYTECODE=1
export VLLM_CACHE_ROOT="$smoke_tmp/vllm" TORCHINDUCTOR_CACHE_DIR="$smoke_tmp/inductor"
export TRITON_CACHE_DIR="$smoke_tmp/triton" MPLCONFIGDIR="$smoke_tmp/matplotlib"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export CUDA_VISIBLE_DEVICES="$gpu" CUDA_HOME=/usr/local/cuda-13.0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$output"
cleanup() {
  rc=$?
  export LD_LIBRARY_PATH="$system_library_path"
  cd /tmp
  rm -r -- "$smoke_tmp"
  sleep 5
  python3 - "$gpu" <<'PY'
import subprocess, sys
gpu = sys.argv[1]
memory = int(subprocess.check_output(['nvidia-smi', '-i', gpu,
    '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
pane = f'{gpu}:0.0'
pid = subprocess.check_output(['tmux', 'display-message', '-p', '-t', pane, '#{pane_pid}'], text=True).strip()
children = subprocess.run(['pgrep', '-P', pid], capture_output=True, text=True)
if memory <= 64 and children.returncode == 1:
    command = ('cd /media/damoxing/tangzecong && BURN_STEPS=2000 BURN_BATCH=8 '
        'BURN_CUTOFF_LEN=4096 BURN_IMAGE_PIXELS=1572864 BURN_LORA_RANK=128 '
        f'bash ./llamafactory_burn.sh {gpu}')
    subprocess.run(['tmux', 'send-keys', '-t', pane, command, 'C-m'], check=True)
    print(f'GPU {gpu}: burn start sent to idle pane')
else:
    print(f'GPU {gpu}: restore skipped; memory={memory}, pane occupied={children.returncode == 0}')
PY
  exit "$rc"
}
trap cleanup EXIT
cd "$smoke_tmp"
for mode in ${EDIVAL_SMOKE_COMPONENTS:-metrics judge hps}; do
  case "$mode" in
    metrics) name=EdiVal ;;
    judge) name=EdiVal-judge ;;
    hps) name=EdiVal-hps ;;
    *) echo "Unknown component: $mode" >&2; exit 2 ;;
  esac
  for attempt in {1..30}; do
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
    (( used <= 64 )) && break
    sleep 1
  done
  (( used <= 64 )) || { echo "GPU became occupied before $mode: $used MiB" >&2; exit 1; }
  export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:/home/chs/conda/envs/$name/lib:$system_library_path"
  "/home/chs/conda/envs/$name/bin/python" "$project_dir/scripts/smoke_edival_scoring_environment.py" \
    --mode "$mode" --output "$output/$mode-smoke.json" "$@"
done

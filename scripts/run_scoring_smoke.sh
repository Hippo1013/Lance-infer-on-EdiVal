#!/usr/bin/env bash
# GPU must already be released using the workspace burn procedure.
set -euo pipefail
gpu="${1:?GPU index required}"; shift
mode="${1:?metrics or judge required}"; shift
[[ "$gpu" == 0 || "$gpu" == 1 ]] || exit 2
case "$mode" in
  metrics) python_exe=/home/chs/conda/envs/mice-metrics/bin/python ;;
  judge) python_exe=/home/chs/conda/envs/mice-judge/bin/python ;;
  *) exit 2 ;;
esac
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
if (( used > 64 )); then echo "GPU $gpu is occupied ($used MiB); release it first" >&2; exit 1; fi
smoke_tmp="$(mktemp -d /tmp/umm-scoring-smoke.XXXXXX)"
system_library_path="${LD_LIBRARY_PATH:-}"
export TMPDIR="$smoke_tmp" XDG_CACHE_HOME="$smoke_tmp/cache"
export VLLM_CACHE_ROOT="$smoke_tmp/vllm" TORCHINDUCTOR_CACHE_DIR="$smoke_tmp/inductor"
export TRITON_CACHE_DIR="$smoke_tmp/triton" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
export CUDA_VISIBLE_DEVICES="$gpu" CUDA_HOME=/usr/local/cuda-13.0
env_dir="$(dirname -- "$(dirname -- "$python_exe")")"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:$env_dir/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cleanup() {
  rc=$?
  # Conda's libtinfo must not shadow the system tmux/ncurses runtime.
  export LD_LIBRARY_PATH="$system_library_path"
  unset TMPDIR XDG_CACHE_HOME VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
  rm -r -- "$smoke_tmp"
  sleep 5
  # Restore this run's physical GPU only, so another smoke run can use its card.
  /home/chs/conda/envs/lance/bin/python - "$gpu" <<'PY'
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
    print(f'GPU {gpu}: burn restore skipped; GPU or pane still occupied')
PY
  exit "$rc"
}
trap cleanup EXIT
"$python_exe" "$project_dir/scripts/smoke_scoring_environment.py" --mode "$mode" "$@"

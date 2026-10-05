#!/usr/bin/env bash
set -euo pipefail
edival_gpu="${1:?GPU index required}"
edival_output="${2:?Output directory required}"
shift 2
[[ "$edival_gpu" == 0 || "$edival_gpu" == 1 ]] || exit 2
edival_runtime="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
edival_python=/home/chs/conda/envs/lance/bin/python
edival_temp="$(mktemp -d /tmp/umm-edival-full.XXXXXX)"
edival_library_path="${LD_LIBRARY_PATH:-}"
export TMPDIR="$edival_temp" XDG_CACHE_HOME="$edival_temp/cache"
export VLLM_CACHE_ROOT="$edival_temp/vllm" TORCHINDUCTOR_CACHE_DIR="$edival_temp/inductor"
export TRITON_CACHE_DIR="$edival_temp/triton" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$edival_runtime/src" CUDA_VISIBLE_DEVICES="$edival_gpu" CUDA_HOME=/usr/local/cuda-13.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:/home/chs/conda/envs/lance/lib:$edival_library_path"
cleanup() {
  edival_rc=$?
  export LD_LIBRARY_PATH="$edival_library_path"
  rm -r -- "$edival_temp"
  sleep 5
  "$edival_python" -B - "$edival_gpu" "$edival_output" "$edival_runtime" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[3])/'scripts'))
from full_edival_job import restore_burn
result=restore_burn(sys.argv[1])
output=Path(sys.argv[2])
if output.is_dir():(output/'burn_restore.json').write_text(json.dumps(result)+'\n')
print(json.dumps(result),flush=True)
PY
  exit "$edival_rc"
}
trap cleanup EXIT
"$edival_python" -B "$edival_runtime/scripts/full_edival_job.py" \
  --gpu "$edival_gpu" --output "$edival_output" "$@"

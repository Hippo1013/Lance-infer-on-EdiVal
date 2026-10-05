#!/usr/bin/env bash
# Run on the server only after it is available. No host driver changes.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep this project's environment separate from the shared Conda environments.
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/home/chs/conda/envs}"
export PIP_INDEX_URL="${LANCE_PIP_INDEX_URL:-https://pypi.org/simple}"
conda_exe="${CONDA_EXE:-}"
if [[ -z "$conda_exe" ]]; then conda_exe="$(command -v conda || true)"; fi
if [[ -z "$conda_exe" || ! -x "$conda_exe" ]]; then
  echo 'Conda not found. Inspect existing installations, then set CONDA_EXE to the executable.' >&2
  exit 1
fi
lance_env_path() {
  local envs_json
  envs_json="$("$conda_exe" env list --json)"
  CONDA_ENVS_JSON="$envs_json" python3 - <<'PY'
import json, os, pathlib
matches=[p for p in json.loads(os.environ['CONDA_ENVS_JSON'])['envs'] if pathlib.Path(p).name=='lance']
if len(matches)>1: raise SystemExit('Multiple lance environments: select the intended Conda installation.')
print(matches[0] if matches else '')
PY
}
env_path="$(lance_env_path)"
if [[ -z "$env_path" ]]; then
  "$conda_exe" create -y -n lance --override-channels -c conda-forge python=3.12 pip
  env_path="$(lance_env_path)"
fi
if [[ -z "$env_path" ]]; then echo 'Could not locate the created lance environment.' >&2; exit 1; fi
"$conda_exe" run --no-capture-output -p "$env_path" python -c 'import sys; assert sys.version_info[:2] == (3,12), "lance must use Python 3.12"'
# Install matching binary dependencies together; do not mix with the native
# Lance torch 2.5 stack or the server's unrelated default Python environment.
omni_spec='vllm-omni @ https://github.com/vllm-project/vllm-omni/archive/b742f86136d28423903a1213adde2a62a11067a2.tar.gz'
if [[ -n "${LANCE_OMNI_SOURCE_ARCHIVE:-}" ]]; then
  # Optional transport for a git archive of the exact same commit. Include
  # every vllm_omni/ and requirements/ file; omit only non-packaged demo/tests.
  omni_spec="$(python3 - <<'PY'
import hashlib, os, pathlib, tarfile
rev = 'b742f86136d28423903a1213adde2a62a11067a2'
p = pathlib.Path(os.environ['LANCE_OMNI_SOURCE_ARCHIVE']).resolve(strict=True)
expected = os.environ.get('LANCE_OMNI_SOURCE_SHA256')
if not expected or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
    raise SystemExit('Source archive transfer checksum mismatch')
if p.name != f'vllm-omni-{rev}.tar.gz':
    raise SystemExit('Source archive filename must include the pinned revision')
with tarfile.open(p) as archive:
    if archive.pax_headers.get('comment') != rev:
        raise SystemExit('Source archive is not exported from the pinned Git commit')
print('vllm-omni @ ' + p.as_uri())
PY
)"
fi
# GitHub's source archive has no Git tags. Supply an explicit local version
# rather than accepting the upstream invalid "dev" fallback. Source identity
# is checked separately through the immutable direct URL in preflight.py.
VLLM_OMNI_TARGET_DEVICE=cuda VLLM_OMNI_VERSION_OVERRIDE=0.0.0+gb742f86136d2 \
"$conda_exe" run --no-capture-output -p "$env_path" python -m pip install \
  'vllm==0.30.0' \
  "$omni_spec"
"$conda_exe" run --no-capture-output -p "$env_path" python -m pip install -e "$project_dir"
"$conda_exe" run --no-capture-output -p "$env_path" python -m pip check
echo 'Environment prepared. Run the documented preflight and single-turn checks before a benchmark.'

"""Prepare exact official EdiVal weights, with file and upstream hash verification."""
import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import threading
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=Path, default=Path("/home/chs/model"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    config_path = Path(__file__).resolve().parents[1] / "configs/edival_scoring_models.json"
    config = json.loads(config_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    state = {"status": "running", "pid": os.getpid(), "models": {}}

    def update(name=None, **fields):
        with lock:
            if name:
                state["models"].setdefault(name, {}).update(fields)
            else:
                state.update(fields)
            state["updated_at"] = time.time()
            temp = args.output / "downloads.json.tmp"
            temp.write_text(json.dumps(state, indent=2) + "\n")
            temp.replace(args.output / "downloads.json")

    def prepare(spec):
        name, repo, revision = (spec[k] for k in ("name", "repo", "revision"))
        dest = args.model_root / name
        dest.mkdir(parents=True, exist_ok=True)
        update(name, status="listing", repo=repo, revision=revision)
        info = HfApi().model_info(repo, revision=revision, files_metadata=True, timeout=60)
        files = [f for f in info.siblings if "/" not in f.rfilename and
                 (fnmatch.fnmatch(f.rfilename, spec["weights"]) or
                  f.rfilename.endswith((".json", ".txt", ".model", ".jinja", ".md")) or
                  f.rfilename.startswith("LICENSE"))]
        if not any(fnmatch.fnmatch(f.rfilename, spec["weights"]) for f in files):
            raise ValueError("Weight pattern does not match any upstream file")
        update(name, status="verifying" if spec.get("existing_only") else "downloading",
               total_files=len(files), total_bytes=sum(f.size or 0 for f in files))
        records = []
        for f in files:
            path = dest / f.rfilename
            if spec.get("existing_only"):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing previously verified local file: {path}")
            else:
                # Hub download resumes incomplete transfers. Local files are verified below.
                path = Path(hf_hub_download(repo, f.rfilename, revision=revision, local_dir=dest))
            if f.size is not None and path.stat().st_size != f.size:
                raise ValueError(f"Size mismatch: {name}/{f.rfilename}")
            sha = digest(path)
            upstream = f.lfs.sha256 if f.lfs else None
            if upstream and sha != upstream:
                raise ValueError(f"Upstream SHA256 mismatch: {name}/{f.rfilename}")
            tensors = None
            if path.suffix == ".safetensors":
                with safe_open(path, framework="pt", device="cpu") as sf:
                    tensors = len(sf.keys())
            records.append({"file": f.rfilename, "bytes": path.stat().st_size,
                            "sha256": sha, "upstream_sha256": upstream, "tensor_count": tensors})
            update(name, verified_files=len(records), verified_bytes=sum(r["bytes"] for r in records))
        index = dest / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            if not set(weight_map.values()).issubset({r["file"] for r in records}):
                raise ValueError("Incomplete safetensors shard index")
            tensor_keys = set()
            for shard in set(weight_map.values()):
                with safe_open(dest / shard, framework="pt", device="cpu") as sf:
                    tensor_keys.update(sf.keys())
            if set(weight_map) != tensor_keys:
                raise ValueError("Shard tensor keys differ from index")
        manifest = {**spec, "directory": str(dest), "files": records}
        (args.output / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        update(name, status="verified")

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(prepare, spec): spec["name"] for spec in config["models"]}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                failed.append(name)
                update(name, status="failed", error_type=type(exc).__name__, error=str(exc)[:500])
    update(status="failed" if failed else "verified", failed=failed)
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()

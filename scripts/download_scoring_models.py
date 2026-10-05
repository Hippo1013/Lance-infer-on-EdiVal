"""Download pinned scoring resources; verify published LFS hashes and safetensors.

Requires proxy variables to be set by the caller. Does not use GPUs.
Manifests and progress are installation evidence, separate from capability scores.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
from huggingface_hub import HfApi, hf_hub_download
import requests
from safetensors import safe_open


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=Path, default=Path("/home/chs/model"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("https_proxy"):
        raise SystemExit("Set the approved server proxy before downloading")
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/scoring_models.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    state = {"status": "running", "models": {}, "pid": os.getpid()}

    def update(name, **fields):
        with lock:
            state["models"].setdefault(name, {}).update(fields)
            state["updated_at"] = time.time()
            tmp = args.output / "downloads.json.tmp"
            tmp.write_text(json.dumps(state, indent=2) + "\n")
            tmp.replace(args.output / "downloads.json")

    def download_one(spec):
        name, repo, revision = spec["name"], spec["repo"], spec["revision"]
        dest = args.model_root / name
        dest.mkdir(parents=True, exist_ok=True)
        update(name, status="listing", repo=repo, revision=revision)
        info = HfApi().model_info(repo, revision=revision, files_metadata=True, timeout=60)
        files = [f for f in info.siblings if "/" not in f.rfilename and
                 (f.rfilename.endswith((".safetensors", ".json", ".txt", ".model", ".jinja", ".md")) or
                  f.rfilename.startswith("LICENSE"))]
        update(name, status="downloading", total_bytes=sum(f.size or 0 for f in files), files=len(files))

        def fetch(f):
            for attempt in range(4):
                try:
                    path = Path(hf_hub_download(repo, f.rfilename, revision=revision, local_dir=dest))
                    if f.size is not None and path.stat().st_size != f.size:
                        raise ValueError("size mismatch")
                    sha = digest(path)
                    upstream = f.lfs.sha256 if f.lfs else None
                    if upstream and sha != upstream:
                        raise ValueError("LFS SHA256 mismatch")
                    tensors = None
                    if path.suffix == ".safetensors":
                        with safe_open(path, framework="pt", device="cpu") as sf:
                            tensors = len(sf.keys())
                    return {"file": f.rfilename, "bytes": path.stat().st_size,
                            "sha256": sha, "upstream_sha256": upstream, "tensor_count": tensors}
                except Exception as exc:
                    update(name, last_error=type(exc).__name__, attempt=attempt + 1)
                    if attempt == 3:
                        raise
                    time.sleep(3 * (attempt + 1))

        records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(fetch, f) for f in files]
            for future in concurrent.futures.as_completed(futures):
                records.append(future.result())
                update(name, verified_files=len(records), verified_bytes=sum(r["bytes"] for r in records))
        index = dest / "model.safetensors.index.json"
        if index.exists():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if not shards.issubset({r["file"] for r in records}):
                raise ValueError("incomplete shard index")
        (args.output / f"{name}.manifest.json").write_text(json.dumps({**spec, "files": records}, indent=2) + "\n")
        update(name, status="verified")

    def grounding():
        name = "GroundingDINO-SwinT-OGC"
        dest = args.model_root / name
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "groundingdino_swint_ogc.pth"
        url = cfg["groundingdino"]["weights_url"]
        update(name, status="downloading")
        # No published checksum is assumed for this release asset.
        if not path.exists():
            part = path.with_suffix(".pth.partial")
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                expected = int(response.headers.get("content-length", 0))
                with part.open("wb") as out:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        out.write(chunk)
                if expected and part.stat().st_size != expected:
                    raise ValueError("GroundingDINO download size mismatch")
            part.replace(path)
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "model" not in checkpoint:
            raise ValueError("GroundingDINO state dictionary missing")
        record = {"url": url, "bytes": path.stat().st_size, "sha256": digest(path),
                  "upstream_sha256": None, "state_tensors": len(checkpoint["model"])}
        (args.output / f"{name}.manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        update(name, status="verified", **record)

    jobs = [("GroundingDINO-SwinT-OGC", grounding)] + [(s["name"], lambda s=s: download_one(s)) for s in cfg["models"]]
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): name for name, fn in jobs}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                failed.append(name)
                update(name, status="failed", error_type=type(exc).__name__)
    state["status"] = "failed" if failed else "verified"
    update("_summary", failed=failed)
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()

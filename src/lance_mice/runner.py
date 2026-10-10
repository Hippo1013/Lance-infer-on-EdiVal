"""Multi-turn image-editing driver with dry-run, whole-session sharding, and verified resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image

from .dataset import DEFAULT_ROOTS, load_samples, shard_samples
from .images import image_hash
from .protocol import digest, protocol_manifest
from .settings import Settings


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".part")
    staging.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staging.replace(path)


def model_identity(model: Path) -> dict:
    names = ("Lance_3B/model.safetensors", "Lance_3B/llm_config.json",
             "Qwen2.5-VL-ViT/vit.safetensors", "Wan2.2_VAE.pth")
    identity = {}
    for name in names:
        path = model / name
        stat = path.stat()
        identity[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return identity


def run_session(sample, output: Path, settings: Settings, backend, *, run_id: str,
                resume=False, max_turns=3, save_attention=False):
    directory = output / sample.split / sample.sample_id
    directory.mkdir(parents=True, exist_ok=True)
    source = sample.read_source()
    fingerprint = digest([sample.session_id, sample.instructions, image_hash(source),
                          settings.identity(), run_id])
    spec = {"fingerprint": fingerprint, "session_id": sample.session_id,
            "source": str(sample.image), "source_hash": image_hash(source),
            "instructions": list(sample.instructions), "settings": settings.identity()}
    if sample.image_member is not None:
        spec["source_member"] = sample.image_member
    spec_path = directory / "session.json"
    if spec_path.exists():
        if not resume:
            raise FileExistsError(f"Existing session; use --resume: {directory}")
        if json.loads(spec_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError(f"Resume configuration or source changed: {directory}")
        with Image.open(directory / "turn_0_input.png") as im:
            if image_hash(im) != spec["source_hash"]:
                raise ValueError(f"Saved source image was changed: {directory}")
    else:
        if any(directory.iterdir()):
            raise ValueError(f"Unrecognized existing session contents: {directory}")
        write_json(spec_path, spec)
        source.save(directory / "turn_0_input.png")
    # Refuse gaps, orphan files and tampered outputs instead of silently overwriting.
    total_turns = len(sample.instructions)
    final_turn = min(max_turns, total_turns)
    if final_turn < 1:
        raise ValueError("max_turns must be positive")
    allowed = {"turn_0_input.png"} | {f"turn_{t}.{ext}" for t in range(1, total_turns + 1)
                                   for ext in (("png", "json", "attention.npz") if save_attention else ("png", "json"))}
    unexpected = {p.name for p in directory.glob("turn_*")} - allowed
    if unexpected:
        raise ValueError(f"Unexpected saved turns: {sorted(unexpected)}")
    complete = 0
    for turn in range(1, total_turns + 1):
        png, meta = directory / f"turn_{turn}.png", directory / f"turn_{turn}.json"
        attention_file = directory / f"turn_{turn}.attention.npz"
        if save_attention and attention_file.exists() != png.exists():
            raise ValueError(f"Incomplete attention output for turn {turn}: {directory}")
        if png.exists() != meta.exists():
            raise ValueError(f"Incomplete output pair for turn {turn}: {directory}")
        if png.exists():
            if turn != complete + 1:
                raise ValueError(f"Gap in saved history: {directory}")
            complete = turn
    images = [source]
    rows = []
    for turn, instruction in enumerate(sample.instructions, 1):
        if turn > max_turns:
            break
        png, meta_path = directory / f"turn_{turn}.png", directory / f"turn_{turn}.json"
        inputs = [image_hash(im) for im in images]
        if turn <= complete:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if save_attention:
                attention_file = directory / f"turn_{turn}.attention.npz"
                if hashlib.sha256(attention_file.read_bytes()).hexdigest() != meta["backend"]["attention"]["sha256"]:
                    raise ValueError(f"Saved attention was changed: {attention_file}")
            with Image.open(png) as im:
                result = im.convert("RGB")
            if (meta["session_fingerprint"] != fingerprint or meta["input_hashes"] != inputs
                    or meta["output_hash"] != image_hash(result)):
                raise ValueError(f"Saved history verification failed: {png}")
        else:
            extra = {}
            if save_attention:
                attention_staged = directory / f"turn_{turn}.attention.npz.part"
                extra["attention_path"] = attention_staged
            start = time.perf_counter()
            result, diagnostics = backend.edit(sample.session_id, images,
                list(sample.instructions[:turn]), end_session=(turn == final_turn), **extra)
            elapsed = time.perf_counter() - start
            if not isinstance(result, Image.Image):
                raise TypeError("Backend did not return a PIL image")
            result = result.convert("RGB")
            meta = {"session_fingerprint": fingerprint, "turn": turn,
                    "instruction": instruction, "input_hashes": inputs,
                    "output_hash": image_hash(result), "wall_seconds": elapsed,
                    "protocol": protocol_manifest(list(sample.instructions[:turn]), version=settings.history_protocol),
                    "backend": diagnostics}
            staged_png = png.with_suffix(".png.part")
            result.save(staged_png, format="PNG")
            staged_png.replace(png)
            if save_attention:
                if hashlib.sha256(attention_staged.read_bytes()).hexdigest() != diagnostics["attention"]["sha256"]:
                    raise ValueError("Attention artifact transport mismatch")
                attention_staged.replace(directory / f"turn_{turn}.attention.npz")
            write_json(meta_path, meta)
        rows.append(meta)
        # Feed exactly the saved RGB pixels, including after a resumed run.
        with Image.open(png) as im:
            images.append(im.convert("RGB"))
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=("mice", "imgedit", "edival"), default="mice")
    p.add_argument("--dataset", type=Path)
    p.add_argument("--model", type=Path, default=Path("/home/chs/model/Lance"))
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--selection", choices=("smoke", "all"), default="smoke")
    p.add_argument("--profile", choices=("single", "dp2", "cfg2"), default="single")
    p.add_argument("--gpus", default="0")
    p.add_argument("--cache-mode", choices=("none", "images", "prefix"))
    p.add_argument("--history-protocol", choices=("lance-history-bare-v2", "lance-history-chat-v1"))
    p.add_argument("--attention-format", choices=("target-group-mass-v1", "target-token-region-stats-v1"))
    p.add_argument("--resolution", type=int, choices=(512, 768))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--save-attention", action="store_true", help="Save per-step/layer/head target-to-context group attention")
    p.add_argument("--max-turns", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--worker-index", type=int, default=-1, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.save_attention and args.profile == "cfg2":
        raise ValueError("Attention observation supports single or DP2 only")
    settings = Settings.from_file(args.config) if args.config else Settings()
    if args.resolution:
        settings = replace(settings, resolution=args.resolution)
    if args.cache_mode:
        settings = replace(settings, cache_mode=args.cache_mode)
    if args.history_protocol:
        settings = replace(settings, history_protocol=args.history_protocol)
    if args.attention_format:
        settings = replace(settings, attention_format=args.attention_format)
    args.dataset = args.dataset or DEFAULT_ROOTS[args.benchmark]
    samples = load_samples(args.dataset, args.selection, args.benchmark)
    workers = 2 if args.profile == "dp2" else 1
    if args.dry_run:
        print(json.dumps({"sessions": len(samples), "benchmark": args.benchmark, "turns": sum(min(args.max_turns, len(s.instructions)) for s in samples),
            "splits": {k: sum(s.split == k for s in samples) for k in sorted({s.split for s in samples})},
            "assignments": [[s.session_id for s in shard_samples(samples, i, workers)] for i in range(workers)],
            "settings": settings.identity()}, ensure_ascii=False, indent=2))
        return 0
    if args.output is None:
        raise ValueError("--output is required for inference")
    gpus = args.gpus.split(",")
    expected = 1 if args.profile == "single" else 2
    if len(gpus) != expected or len(set(gpus)) != expected or any(not x.strip() for x in gpus):
        raise ValueError(f"{args.profile} needs {expected} distinct GPU identifiers")
    if args.worker_index not in (-1, 0, 1) or (workers == 1 and args.worker_index != -1):
        raise ValueError("Invalid internal worker index")

    run_spec = {"settings": settings.identity(), "profile": args.profile, "audit": args.audit,
                "model": str(args.model.resolve()), "model_files": model_identity(args.model),
                "selection": args.selection,
                "samples": [(s.session_id, s.instructions) for s in samples]}
    if args.benchmark != "mice":
        run_spec["benchmark"] = args.benchmark
    if args.benchmark == "edival":
        from .edival_dataset import source_identity
        run_spec["dataset_source"] = source_identity(args.dataset)
    if args.save_attention:
        from .attention import ATTENTION_VERSION
        run_spec["attention"] = settings.attention_format
    run_id = digest(run_spec)
    manifest = args.output / "run.json"
    if args.worker_index == -1:
        if manifest.exists():
            if not args.resume or json.loads(manifest.read_text())["fingerprint"] != run_id:
                raise ValueError("Output already exists or run configuration differs")
        else:
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError("Output contains unrelated files")
            write_json(manifest, {"fingerprint": run_id, **run_spec})
    elif not manifest.exists() or json.loads(manifest.read_text())["fingerprint"] != run_id:
        raise ValueError("Worker manifest is missing or differs from coordinator")

    if args.profile == "dp2" and args.worker_index == -1:
        children = []
        start = time.perf_counter()
        try:
            child_args = list(sys.argv[1:] if argv is None else argv)
            for worker in range(2):
                cmd = [sys.executable, "-m", "lance_mice.runner", *child_args, "--worker-index", str(worker)]
                children.append(subprocess.Popen(cmd))
            statuses = [child.wait() for child in children]
            if any(statuses):
                raise RuntimeError(f"Worker failures: {statuses}")
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
        write_json(args.output / "timing.json", {"profile": "dp2", "wall_seconds": time.perf_counter() - start,
                                                  "includes_model_load": True})
        return 0

    worker = max(args.worker_index, 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[worker] if workers == 2 else args.gpus
    from .backend import OmniBackend
    start = time.perf_counter()
    backend = OmniBackend(args.model, settings, cfg_parallel_size=2 if args.profile == "cfg2" else 1,
                          audit=args.audit)
    load_seconds = time.perf_counter() - start
    rows = []
    start = time.perf_counter()
    try:
        for sample in shard_samples(samples, worker, workers):
            rows.extend(run_session(sample, args.output, settings, backend, run_id=run_id,
                                    resume=args.resume, max_turns=args.max_turns, save_attention=args.save_attention))
    finally:
        backend.close()
    write_json(args.output / f"worker_{worker}.json", {
        "worker": worker, "model_load_seconds": load_seconds, "session_seconds": time.perf_counter() - start,
        "turns": len(rows), "generated_turns": backend.generated_turns,
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "sessions": [s.session_id for s in shard_samples(samples, worker, workers)],
        "pid": os.getpid(), "completed": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

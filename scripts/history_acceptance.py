"""Explicit single-GPU acceptance for full history, cache and cold resume.

Run only after the same-conditioning single-turn probe has passed. Does not
run DP/CFG parallelism, throughput tests, all-session inference or scoring.
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from PIL import Image

from lance_mice.acceptance import compare_runs, validate_run
from lance_mice.backend import OmniBackend
from lance_mice.dataset import load_samples
from lance_mice.protocol import digest
from lance_mice.runner import model_identity, run_session, write_json
from lance_mice.settings import Settings


def prepare_run(root, model, settings, samples):
    spec = {"settings": settings.identity(), "profile": "single", "audit": True,
            "model": str(model.resolve()), "model_files": model_identity(model),
            "selection": "smoke", "samples": [(s.session_id, s.instructions) for s in samples]}
    run_id = digest(spec)
    if root.exists():
        raise FileExistsError(root)
    write_json(root / "run.json", {"fingerprint": run_id, **spec})
    return run_id


class NoInference:
    def edit(self, *args, **kwargs):
        raise AssertionError("Invalid checkpoint reached inference")


def recovery_guards(root, sample, settings, run_id):
    results = []
    # Copies used for deliberate damage are disposable system-temp artifacts.
    with tempfile.TemporaryDirectory(prefix="lance-recovery-guards-") as tmp:
        for case in ("changed_settings", "missing_pair", "tampered_output", "history_gap"):
            copied = Path(tmp) / case
            folder = copied / sample.session_id
            shutil.copytree(root / sample.session_id, folder)
            test_settings = settings
            if case == "changed_settings":
                test_settings = replace(settings, seed=settings.seed + 1)
            elif case == "missing_pair":
                (folder / "turn_1.json").unlink()
            elif case == "tampered_output":
                with Image.open(folder / "turn_1.png") as image:
                    image = image.convert("RGB")
                pixel = image.getpixel((0, 0))
                image.putpixel((0, 0), ((pixel[0] + 1) % 256, pixel[1], pixel[2]))
                image.save(folder / "turn_1.png")
            else:
                (folder / "turn_1.png").unlink()
                (folder / "turn_1.json").unlink()
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            try:
                run_session(sample, copied, test_settings, NoInference(), run_id=run_id, resume=True)
            except ValueError as error:
                after = {p.name: p.read_bytes() for p in folder.iterdir()}
                if before != after:
                    raise AssertionError(f"Guard changed checkpoint files: {case}")
                results.append({"case": case, "status": "passed", "error": str(error)})
            else:
                raise AssertionError(f"Invalid checkpoint accepted: {case}")
    return results


def cold_resume(args, samples, settings, resumed, run_id, saved):
    # A new interpreter guarantees that no Python/model/KV/RNG state survives.
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    subprocess.run([sys.executable, "-m", "lance_mice.runner", "--model", str(args.model),
                    "--dataset", str(args.dataset), "--profile", "single", "--gpus", args.gpu,
                    "--audit", "--resume", "--output", str(resumed)], check=True)
    worker = json.loads((resumed / "worker_0.json").read_text())
    expected = [[s.session_id, t] for s in samples for t in (2, 3)]
    if worker["pid"] == os.getpid() or worker["generated_turns"] != expected:
        raise AssertionError(f"Unexpected process or generated turns: {worker}")
    if any((p.read_bytes(), p.stat().st_mtime_ns) != value for p, value in saved.items()):
        raise AssertionError("Resume rewrote the completed first turn")
    comparison = compare_runs(args.output / "smoke_none", resumed)
    write_json(args.output / f"{resumed.name}_comparison.json", comparison)
    if comparison["status"] != "passed":
        raise AssertionError("Cold resume differs from uninterrupted inference")
    guards = recovery_guards(resumed, samples[0], settings, run_id)
    result = {"status": "passed", "parent_pid": os.getpid(), "resume_pid": worker["pid"],
              "generated_turns": worker["generated_turns"], "first_turn_unchanged": True, "guards": guards}
    write_json(resumed / "recovery.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/chs/model/Lance"))
    parser.add_argument("--dataset", type=Path, default=Path("/home/chs/dataset/MICE-Bench"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--single-turn-reference", type=Path)
    parser.add_argument("--recovery-only", action="store_true",
                        help="Run a fresh-process recovery check against an existing smoke_none baseline")
    args = parser.parse_args()
    reference = args.single_turn_reference or args.output / "single_turn/comparison.json"
    if json.loads(reference.read_text())["status"] != "passed":
        raise ValueError("Single-turn parity must pass first")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    samples = load_samples(args.dataset)
    settings = Settings()
    if args.recovery_only:
        resumed = args.output / "smoke_resume_process"
        run_id = prepare_run(resumed, args.model, settings, samples)
        backend = OmniBackend(args.model, settings, audit=True)
        try:
            for sample in samples:
                run_session(sample, resumed, settings, backend, run_id=run_id, max_turns=1)
            saved = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                     for sample in samples for p in (resumed / sample.session_id).glob("turn_1.*")}
        finally:
            backend.close()
        del backend
        cold_resume(args, samples, settings, resumed, run_id, saved)
        print("Fresh-process recovery acceptance passed", flush=True)
        return
    report = {"status": "running", "scope": "steps 3-5, single GPU", "stages": {}}
    backend = OmniBackend(args.model, settings, audit=True)
    try:
        for mode in ("none", "images", "prefix"):
            current = replace(settings, cache_mode=mode)
            backend.settings = current
            root = args.output / f"smoke_{mode}"
            run_id = prepare_run(root, args.model, current, samples)
            for sample in samples:
                print(f"START {mode} {sample.session_id}", flush=True)
                run_session(sample, root, current, backend, run_id=run_id)
            rows = validate_run(root)
            write_json(root / "validation.json", {"status": "passed", "turns": len(rows)})
            if mode != "none":
                comparison = compare_runs(args.output / "smoke_none", root)
                write_json(args.output / f"{mode}_comparison.json", comparison)
                if comparison["status"] != "passed":
                    raise AssertionError(f"Cache differs from none: {mode}")
            report["stages"][mode] = {"status": "passed", "turns": len(rows)}
            write_json(args.output / "history_summary.json", report)
        backend.settings = settings
        resumed = args.output / "smoke_resume"
        run_id = prepare_run(resumed, args.model, settings, samples)
        for sample in samples:
            run_session(sample, resumed, settings, backend, run_id=run_id, max_turns=1)
        saved = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                 for sample in samples for p in (resumed / sample.session_id).glob("turn_1.*")}
    finally:
        backend.close()
    del backend
    report["stages"]["resume"] = cold_resume(args, samples, settings, resumed, run_id, saved)
    report["status"] = "passed"
    write_json(args.output / "history_summary.json", report)
    print("History/cache/recovery acceptance passed", flush=True)


if __name__ == "__main__":
    main()

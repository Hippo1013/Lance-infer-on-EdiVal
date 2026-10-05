"""Warm one persistent model per GPU and time fixed CM/CU batches three times.

Run after DP2 correctness acceptance. Loading and each mode's complete warmup
are excluded from measurements. Images and timing records are durable outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from lance_mice.acceptance import compare_runs, validate_run
from lance_mice.dataset import load_samples, shard_samples
from lance_mice.performance import batch_measurement, summarize_mode
from lance_mice.protocol import digest
from lance_mice.runner import model_identity, run_session, write_json
from lance_mice.settings import Settings


def read(path):
    return json.loads(path.read_text())


def wait_files(paths, children, timeout=1200):
    deadline = time.monotonic() + timeout
    while not all(p.exists() for p in paths):
        # The faster DP worker can exit normally after writing its final file
        # while its peer is still finishing. Only an absent own result is fatal.
        failures = [child.returncode for path, child in zip(paths, children)
                    if child.poll() is not None and not path.exists()]
        if failures:
            raise RuntimeError(f"Worker exited before completion: {failures}; inspect worker logs")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {paths}")
        time.sleep(0.05)
    return [read(p) for p in paths]


def worker(args):
    # CUDA visibility is established by the coordinator before imports/spawn.
    from lance_mice.backend import OmniBackend
    samples = shard_samples(load_samples(args.dataset), args.worker_index,
                            2 if args.profile == "dp2" else 1)
    start = time.perf_counter()
    backend = OmniBackend(args.model, Settings(), audit=False)
    load_seconds = time.perf_counter() - start
    control = args.output / "control"
    write_json(control / f"ready_{args.worker_index}.json", {
        "pid": os.getpid(), "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "model_load_seconds": load_seconds, "sessions": [s.session_id for s in samples]})
    try:
        for index in range(3 * (args.repetitions + 1)):
            command_path = control / f"phase_{index}.json"
            deadline = time.monotonic() + 1200
            while not command_path.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Coordinator did not issue the next phase")
                time.sleep(0.02)
            command = read(command_path)
            settings = replace(Settings(), cache_mode=command["cache_mode"])
            backend.settings = settings
            backend.generated_turns.clear()
            while time.perf_counter() < command["start_at"]:
                time.sleep(max(0.0, min(0.01, command["start_at"] - time.perf_counter())))
            started = time.perf_counter()
            rows = []
            for sample in samples:
                rows.extend(run_session(sample, Path(command["run"]), settings, backend,
                                        run_id=command["run_id"]))
            finished = time.perf_counter()
            turns = [{"session_id": row["backend"]["session_id"], "turn": row["turn"],
                      "wall_seconds": row["wall_seconds"],
                      "seconds": row["backend"]["seconds"],
                      "counts": row["backend"]["counts"],
                      "prefix_segments_reused": row["backend"]["prefix_segments_reused"],
                      "peak_memory_bytes": row["backend"]["peak_memory_bytes"]} for row in rows]
            write_json(Path(command["run"]) / f"worker_{args.worker_index}.json", {
                "pid": os.getpid(), "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                "sessions": [s.session_id for s in samples], "generated_turns": backend.generated_turns[:],
                "started_at": started, "finished_at": finished, "turns": turns,
                "model_load_seconds": load_seconds, "completed": True})
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/chs/model/Lance"))
    parser.add_argument("--dataset", type=Path, default=Path("/home/chs/dataset/MICE-Bench"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("single", "dp2"), required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--correctness-reference", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker_index >= 0:
        worker(args)
        return
    count = 2 if args.profile == "dp2" else 1
    gpus = args.gpus.split(",")
    if len(gpus) != count or len(set(gpus)) != count or args.repetitions < 3:
        raise ValueError("Use distinct GPUs for this profile and at least three repetitions")
    if args.output.exists():
        raise FileExistsError(args.output)
    validate_run(args.correctness_reference)
    samples = load_samples(args.dataset)
    expected = [s.session_id for s in samples]
    control = args.output / "control"
    control.mkdir(parents=True)
    children, logs = [], []
    load_start = time.perf_counter()
    report = {"status": "running", "profile": args.profile, "gpus": gpus,
              "settings": Settings().identity(), "samples": expected,
              "audit": False, "warmup_batches_per_mode": 1, "modes": {}}
    try:
        for index, gpu in enumerate(gpus):
            log = (args.output / f"worker_{index}.log").open("w")
            logs.append(log)
            command = [sys.executable, str(Path(__file__).resolve()),
                       *sys.argv[1:], "--worker-index", str(index)]
            children.append(subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                             stdout=log, stderr=subprocess.STDOUT))
        ready = wait_files([control / f"ready_{i}.json" for i in range(count)], children)
        report["cold_start_seconds"] = time.perf_counter() - load_start
        report["workers"] = ready
        phase = 0
        for mode in ("none", "images", "prefix"):
            batches = []
            for repetition in range(-1, args.repetitions):
                name = "warmup" if repetition < 0 else f"repeat_{repetition + 1}"
                root = args.output / mode / name
                settings = replace(Settings(), cache_mode=mode)
                spec = {"settings": settings.identity(), "profile": args.profile, "audit": False,
                        "model": str(args.model.resolve()), "model_files": model_identity(args.model),
                        "selection": "smoke", "samples": [(s.session_id, s.instructions) for s in samples]}
                run_id = digest(spec)
                write_json(root / "run.json", {"fingerprint": run_id, **spec})
                command = {"cache_mode": mode, "run": str(root), "run_id": run_id,
                           "start_at": time.perf_counter() + 0.5}
                print(f"START {args.profile} {mode} {name}", flush=True)
                write_json(control / f"phase_{phase}.json", command)
                workers = wait_files([root / f"worker_{i}.json" for i in range(count)], children)
                batch = batch_measurement(workers, command["start_at"], expected)
                if max(batch["start_lag_seconds"]) > 0.1:
                    raise AssertionError("Worker missed synchronized start by more than 100ms")
                comparison = compare_runs(args.correctness_reference, root)
                write_json(root / "comparison.json", comparison)
                if comparison["status"] != "passed":
                    raise AssertionError(f"Timed run changed output pixels: {root}")
                write_json(root / "batch.json", batch)
                if repetition >= 0:
                    batches.append(batch)
                phase += 1
                print(f"DONE {args.profile} {mode} {name} {batch['batch_seconds']:.3f}s", flush=True)
            report["modes"][mode] = summarize_mode(batches)
            write_json(args.output / "summary.json", report)
        statuses = [child.wait(timeout=60) for child in children]
        if any(statuses):
            raise RuntimeError(f"Worker shutdown failed: {statuses}")
        report["status"] = "passed"
        report["scope"] = ("Two fixed three-turn sessions; same backend and sampling parameters. "
                           "Model loading and warmup excluded. Batch timing includes output I/O. "
                           "DP2 uses twice the GPUs; per-GPU throughput is also reported. "
                           "No native Lance speed or image quality claim.")
        write_json(args.output / "summary.json", report)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            if child.poll() is None:
                child.wait(timeout=60)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()

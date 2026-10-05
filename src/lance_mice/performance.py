"""Summaries for synchronized, warmed inference batches; no CUDA imports."""

from __future__ import annotations

import statistics


def distribution(values):
    values = list(values)
    if not values:
        raise ValueError("Cannot summarize an empty measurement")
    return {"median": statistics.median(values), "mean": statistics.mean(values),
            "min": min(values), "max": max(values), "values": values}


def batch_measurement(workers, start_at, expected_sessions):
    sessions = [s for worker in workers for s in worker["sessions"]]
    if len(sessions) != len(set(sessions)) or sorted(sessions) != sorted(expected_sessions):
        raise ValueError("Worker assignments duplicate or omit complete sessions")
    turns = sum(len(worker["turns"]) for worker in workers)
    if turns != 3 * len(sessions):
        raise ValueError("Incomplete three-turn batch")
    for worker in workers:
        expected = [[s, t] for s in worker["sessions"] for t in (1, 2, 3)]
        if worker["generated_turns"] != expected:
            raise ValueError("Generated turns differ from whole-session assignment")
    elapsed = max(worker["finished_at"] for worker in workers) - start_at
    if elapsed <= 0:
        raise ValueError("Invalid batch clock")
    return {"batch_seconds": elapsed, "sessions": len(sessions), "turns": turns,
            "sessions_per_minute": len(sessions) * 60 / elapsed,
            "turns_per_second": turns / elapsed,
            "sessions_per_gpu_minute": len(sessions) * 60 / elapsed / len(workers),
            "gpu_count": len(workers),
            "start_lag_seconds": [w["started_at"] - start_at for w in workers],
            "workers": workers}


def summarize_mode(batches):
    if len(batches) < 3:
        raise ValueError("At least three warmed repetitions are required")
    turns = [t for b in batches for w in b["workers"] for t in w["turns"]]
    stages = sorted({k for t in turns for k in t["seconds"]})
    per_turn = {}
    for turn in (1, 2, 3):
        rows = [t for t in turns if t["turn"] == turn]
        per_turn[str(turn)] = {
            "edit_wall_seconds": distribution(t["wall_seconds"] for t in rows),
            "stage_seconds": {k: distribution(t["seconds"].get(k, 0.0) for t in rows) for k in stages},
            "peak_memory_gib": distribution(t["peak_memory_bytes"] / 2**30 for t in rows)}
    return {"repetitions": len(batches),
            "batch_seconds": distribution(b["batch_seconds"] for b in batches),
            "sessions_per_minute": distribution(b["sessions_per_minute"] for b in batches),
            "sessions_per_gpu_minute": distribution(b["sessions_per_gpu_minute"] for b in batches),
            "per_turn": per_turn,
            "encoder_counts_per_batch": {
                k: [sum(t["counts"][k] for w in b["workers"] for t in w["turns"]) for b in batches]
                for k in ("vit_encodes", "vae_encodes")},
            "max_start_lag_seconds": max(x for b in batches for x in b["start_lag_seconds"]),
            "peak_memory_gib": max(t["peak_memory_bytes"] for t in turns) / 2**30}

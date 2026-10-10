"""Validate completed inference runs and compare cache outputs (no GPU required)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .images import image_hash
from .protocol import LEGACY_PROTOCOL_VERSION, PROTOCOL_VERSION, CHAT_PROTOCOL_VERSION, derive_seed, protocol_manifest
from .runner import write_json


def validate_run(root: Path, *, sessions=None, validate_attention_files=True) -> dict:
    run = json.loads((root / "run.json").read_text())
    sessions = run["samples"] if sessions is None else sessions
    result = {}
    for session_id, instructions in sessions:
        folder = root / session_id
        previous = []
        with Image.open(folder / "turn_0_input.png") as im:
            previous.append(image_hash(im))
        for turn in range(1, len(instructions) + 1):
            path = folder / f"turn_{turn}.png"
            row = json.loads((folder / f"turn_{turn}.json").read_text())
            with Image.open(path) as im:
                pixels_hash = image_hash(im)
            if row["output_hash"] != pixels_hash or row["input_hashes"] != previous:
                raise ValueError(f"Image history hash mismatch: {path}")
            proto, backend = row["protocol"], row["backend"]
            if (proto["version"] not in {PROTOCOL_VERSION, CHAT_PROTOCOL_VERSION, LEGACY_PROTOCOL_VERSION} or proto["image_count"] != turn
                    or proto["instruction_count"] != turn or proto["target_count"] != 1):
                raise ValueError(f"Protocol mismatch: {path}")
            if proto != protocol_manifest(instructions[:turn], version=proto["version"]):
                raise ValueError(f"Prompt differs from declared protocol: {path}")
            if backend["image_hashes"] != previous or backend["turn"] != turn:
                raise ValueError(f"GPU did not receive the expected history: {path}")
            seed = derive_seed(run["settings"]["settings"]["seed"], session_id, "noise", turn)
            if backend["seed"] != seed:
                raise ValueError(f"Seed mismatch: {path}")
            positive, negative = backend["positive_trace"], backend["negative_trace"]
            images = [x["image_index"] for x in positive if x["kind"] == "vae"]
            vit_images = [x["image_index"] for x in positive if x["kind"] == "vit"]
            if images != list(range(turn)) or vit_images != list(range(turn)):
                raise ValueError(f"Dropped or reordered GPU image blocks: {path}")
            raw = [x for x in positive if x.get("role") in {"history", "current"}]
            if [x["text"] for x in raw] != instructions[:turn]:
                raise ValueError(f"Instruction changed: {path}")
            if len([x for x in raw if x["role"] == "current"]) != 1:
                raise ValueError(f"Current instruction span ambiguous: {path}")
            if any(x.get("role") == "current" for x in negative):
                raise ValueError(f"Current instruction leaked into CFG-negative: {path}")
            guidance = run["settings"]["settings"]["cfg_text_scale"] > 1
            if guidance and [x for x in positive if x.get("role") != "current"] != negative:
                raise ValueError(f"History or labels were removed from CFG-negative: {path}")
            expected_order = []
            for segment in proto["segments"]:
                if segment["kind"] == "image":
                    expected_order.extend([(k, segment["image_index"]) for k in ("vit", "vae")])
                else:
                    expected_order.append(("text", segment["role"], segment["turn"], segment["text"]))
            actual_order = [(x["kind"], x["image_index"]) if x["kind"] in ("vit", "vae")
                else ("text", x["role"], x["turn"], x["text"]) for x in positive]
            if actual_order != expected_order or row["instruction"] != instructions[turn-1]:
                raise ValueError(f"Full interleaving or raw current instruction differs: {path}")
            if run.get("attention") and validate_attention_files:
                observation = backend["attention"]
                if observation["version"] == "target-token-region-stats-v1":
                    from .attention_regions import validate_attention
                else:
                    from .attention import validate_attention
                attention_file = folder / f"turn_{turn}.attention.npz"
                if observation["file"] != attention_file.name or observation["branch"] != "positive":
                    raise ValueError("Attention artifact name or branch differs")
                validate_attention(attention_file, observation, positive,
                    backend["positive_kv_tokens"], run["settings"]["settings"]["steps"])
            previous.append(pixels_hash)
            result[f"{session_id}/turn_{turn}"] = row
    if not result:
        raise ValueError("No completed samples")
    return result


def compare_runs(reference: Path, candidate: Path, *, cfg_comparison=False) -> dict:
    refs, candidates = validate_run(reference), validate_run(candidate)
    if refs.keys() != candidates.keys():
        raise ValueError("Runs contain different cases")
    manifests = [json.loads((root / "run.json").read_text()) for root in (reference, candidate)]
    identities = []
    for manifest in manifests:
        settings = json.loads(json.dumps(manifest["settings"]))
        settings["settings"].pop("cache_mode")
        identities.append((settings, manifest["model_files"], manifest["samples"]))
    if identities[0] != identities[1]:
        raise ValueError("Runs have different model, data or denoising settings")
    diffs = []
    for key in refs:
        left, right = refs[key], candidates[key]
        same = left["output_hash"] == right["output_hash"]
        if left["input_hashes"] != right["input_hashes"] and not cfg_comparison:
            raise ValueError(f"Different conditioning images: {key}")
        with Image.open(reference / f"{key}.png") as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float32)
        with Image.open(candidate / f"{key}.png") as im:
            b = np.asarray(im.convert("RGB"), dtype=np.float32)
        if a.shape != b.shape:
            raise ValueError(f"Output dimensions differ: {key}")
        diffs.append({"case": key, "identical_pixels": same,
                      "mean_absolute_pixel_error_0_255": float(np.abs(a - b).mean())})
    identical = all(x["identical_pixels"] for x in diffs)
    return {"reference": str(reference), "candidate": str(candidate), "turns": len(diffs),
            "identical_pixels": identical,
            "status": ("numeric_comparison_only" if cfg_comparison else "passed" if identical else "failed"),
            "note": ("CFG comparison does not establish task quality or stepwise numerical parity."
                     if cfg_comparison else "Pixel identity is required before enabling cache by default."),
            "differences": diffs}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--cfg-comparison", action="store_true")
    p.add_argument("--report", type=Path)
    args = p.parse_args(argv)
    if args.candidate:
        report = compare_runs(args.reference, args.candidate, cfg_comparison=args.cfg_comparison)
    else:
        rows = validate_run(args.reference)
        report = {"status": "passed", "turns": len(rows), "run": str(args.reference)}
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

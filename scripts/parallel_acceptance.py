"""Acceptance steps 6/7: production DP2 parity and warmed cache/throughput."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from lance_mice.acceptance import compare_runs
from lance_mice.runner import write_json


def read(path):
    return json.loads(path.read_text())


def invoke(root, name, command):
    print(f"START {name}", flush=True)
    with (root / f"{name}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"DONE {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/home/chs/model/Lance"))
    parser.add_argument("--dataset", type=Path, default=Path("/home/chs/dataset/MICE-Bench"))
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    common = ["--model", str(args.model), "--dataset", str(args.dataset)]
    report = {"status": "running", "scope": "Acceptance steps 6/7 only, current history prompt",
              "stages": {}, "quality": "Model capability is an experimental outcome, not an engineering acceptance gate",
              "cfg2": "Optional stepwise CFG comparison not executed"}
    write_json(root / "summary.json", report)
    try:
        for profile, gpus in (("single", "0"), ("dp2", "0,1")):
            invoke(root, f"correctness_{profile}", [sys.executable, "-m", "lance_mice.runner",
                *common, "--profile", profile, "--gpus", gpus, "--audit",
                "--output", str(root / f"correctness_{profile}")])
        comparison = compare_runs(root / "correctness_single", root / "correctness_dp2")
        if comparison["status"] != "passed":
            raise AssertionError("DP2 output differs from single-GPU baseline")
        workers = [read(root / "correctness_dp2" / f"worker_{i}.json") for i in (0, 1)]
        sessions = [s for w in workers for s in w["sessions"]]
        expected = [s[0] for s in read(root / "correctness_single" / "run.json")["samples"]]
        if len(set(sessions)) != len(sessions) or sorted(sessions) != sorted(expected):
            raise AssertionError("DP2 assignments overlap or omit sessions")
        if len({w["pid"] for w in workers}) != 2 or [w["gpu"] for w in workers] != ["0", "1"]:
            raise AssertionError("DP2 does not have distinct workers on GPUs 0/1")
        for w in workers:
            if not w["completed"] or w["generated_turns"] != [[s, t] for s in w["sessions"] for t in (1, 2, 3)]:
                raise AssertionError("DP2 did not generate each assigned turn exactly once")
        report["stages"]["6"] = {"status": "passed", "comparison": comparison, "workers": workers}
        write_json(root / "dp2_comparison.json", report["stages"]["6"])
        write_json(root / "summary.json", report)
        performance = {}
        for profile, gpus in (("single", "0"), ("dp2", "0,1")):
            invoke(root, f"performance_{profile}", [sys.executable,
                str(Path(__file__).with_name("performance_acceptance.py").resolve()), *common,
                "--profile", profile, "--gpus", gpus, "--repetitions", "3",
                "--correctness-reference", str(root / "correctness_single"),
                "--output", str(root / f"performance_{profile}")])
            performance[profile] = read(root / f"performance_{profile}" / "summary.json")
        comparisons = {}
        for mode in ("none", "images", "prefix"):
            single = performance["single"]["modes"][mode]
            dp2 = performance["dp2"]["modes"][mode]
            comparisons[mode] = {
                "single_batch_median_seconds": single["batch_seconds"]["median"],
                "dp2_batch_median_seconds": dp2["batch_seconds"]["median"],
                "dp2_aggregate_throughput_ratio": dp2["sessions_per_minute"]["median"] / single["sessions_per_minute"]["median"],
                "dp2_per_gpu_throughput_ratio": dp2["sessions_per_gpu_minute"]["median"] / single["sessions_per_gpu_minute"]["median"]}
        report["stages"]["7"] = {"status": "passed", "profiles": performance, "comparison": comparisons,
            "note": "Fixed two-session batch, three warmed repetitions. DP2 uses two GPUs; no native-framework speed claim."}
        report["status"] = "technical_checks_passed"
        write_json(root / "summary.json", report)
        print(json.dumps({"status": report["status"], "performance": comparisons}, indent=2), flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        write_json(root / "summary.json", report)
        raise


if __name__ == "__main__":
    main()

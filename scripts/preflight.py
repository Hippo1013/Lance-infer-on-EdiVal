"""Installation inventory, or an explicitly requested CUDA preflight."""

import argparse
import contextlib
import importlib.metadata
import json
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true",
                        help="Read installed package metadata without importing Torch or touching a GPU")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Activate conda environment lance with Python 3.12")
    from lance_mice.protocol import OMNI_REVISION
    dist = importlib.metadata.distribution("vllm-omni")
    direct = dist.read_text("direct_url.json") or ""
    if OMNI_REVISION not in direct:
        raise RuntimeError("vLLM-Omni installation is not the pinned source revision")
    if importlib.metadata.version("vllm").split("+")[0] != "0.30.0":
        raise RuntimeError("Expected vLLM 0.30.0")
    inventory = {"python": platform.python_version(), "environment": sys.prefix,
                 "packages": {name: importlib.metadata.version(name) for name in
                              ("lance-mice", "vllm", "vllm-omni", "torch", "torchvision",
                               "transformers", "diffusers", "safetensors")},
                 "omni_revision": OMNI_REVISION}
    if args.metadata_only:
        print(json.dumps({**inventory, "gpu_checks": "Not started"}, indent=2))
        return
    # Upstream import-time notices belong on stderr; stdout is a JSON report.
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from lance_mice.omni_pipeline import LanceHistoryPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; inspect existing driver/compat libraries")
    checks = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            x = torch.eye(16, device=f"cuda:{index}", dtype=torch.bfloat16)
            if not torch.equal(x @ x, x):
                raise RuntimeError("CUDA BF16 matmul check failed")
            checks.append({"index": index, "name": torch.cuda.get_device_name(index),
                           "capability": torch.cuda.get_device_capability(index), "bf16_matmul": True})
    print(json.dumps({**inventory, "torch": torch.__version__,
                      "cuda_runtime": torch.version.cuda, "gpus": checks,
                      "pipeline_import": LanceHistoryPipeline.__name__,
                      "attention_kernel_and_nccl": "Pending real single/dual GPU inference"}, indent=2))


if __name__ == "__main__":
    main()

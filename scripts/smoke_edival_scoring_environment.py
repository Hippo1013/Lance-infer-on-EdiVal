"""Offline GPU checks of official EdiVal components; never writes capability scores."""
import argparse
import ast
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def redirect_loader(cls, paths):
    original = cls.from_pretrained

    @classmethod
    def local_loader(_cls, name, *args, **kwargs):
        return original(str(paths.get(str(name), name)), *args, **kwargs)

    cls.from_pretrained = local_loader


def metrics(args):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    from groundingdino import _C
    from groundingdino.util.inference import load_model, load_image, predict
    from detector import consistency_detector as cc
    from detector import quality_detector as vq

    models = args.model_root
    mapping = {
        "facebook/dinov3-vitb16-pretrain-lvd1689m": models / "DINOv3-ViT-B-16",
        "google/vit-large-patch16-384": models / "vit-large-patch16-384",
    }
    redirect_loader(AutoImageProcessor, mapping)
    redirect_loader(AutoModel, mapping)
    gd = load_model(str(models / "GroundingDINO-SwinT-OGC/GroundingDINO_SwinT_OGC.local.py"),
                    str(models / "GroundingDINO-SwinT-OGC/groundingdino_swint_ogc.pth"), device="cuda")
    _, image_tensor = load_image(str(args.image))
    boxes, logits, phrases = predict(gd, image_tensor, "painted white leaf pattern",
                                     box_threshold=0.3, text_threshold=0.3, device="cuda")
    assert torch.isfinite(boxes).all() and torch.isfinite(logits).all()
    bundle = cc.load_consistency_model(device="cuda")
    dino, processor = bundle
    source = Image.open(args.image).convert("RGB")
    target = Image.open(args.target).convert("RGB")
    inputs = processor(images=[source, target], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        features = dino(**inputs).last_hidden_state
        same = cc._calculate_dinov3_similarity(bundle, source, source)
        different = cc._calculate_dinov3_similarity(bundle, source, target)
        background = cc._calculate_background_consistency_dino(
            bundle, source, source, mask_img=Image.new("L", source.size, 255))
    assert torch.isfinite(features).all() and features.shape[-1] == 768
    assert abs(same - 1.0) < 1e-5 and math.isfinite(different)
    assert abs(background["bg_dinov3_masked_similarity"] - 1.0) < 1e-5
    quality = vq.RAHF(vit_model=str(models / "vit-large-patch16-384"), t5_model=str(models / "t5-base"))
    quality.load_state_dict(torch.load(models / "RAHF/rahf_model.pt", map_location="cpu", weights_only=True), strict=True)
    quality.eval().to("cuda")
    # HPS has its own environment and check; avoid the official optional lazy loader here.
    vq.load_human_preference_inferencer = lambda device=None: None
    with torch.inference_mode():
        quality_result = vq.evaluate_quality(str(args.target), model=quality)
    assert all(isinstance(v, (float, int)) and math.isfinite(v)
               for k, v in quality_result.items() if k != "human_preference_score")
    return {
        "groundingdino": {"native_extension": str(_C.__file__), "detections": len(phrases),
                          "phrases": phrases, "scores": logits.tolist()},
        "dinov3": {"feature_shape": list(features.shape), "same_image_similarity": same,
                   "source_target_similarity": different, "same_image_background": background},
        "rahf": {"strict_state_load": True, "quality": quality_result},
    }


def judge(args):
    from PIL import Image
    from vllm import LLM, SamplingParams

    # Execute the unchanged upstream image/query helpers, without importing the
    # detection model into the separate vLLM process.
    path = args.official_source / "detector/instruction_detector.py"
    parsed = ast.parse(path.read_text())
    names = {"_image_to_base64", "_query_vlm", "_query_vlm_2_image"}
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    namespace = {"Image": Image, "io": io, "base64": base64, "SamplingParams": SamplingParams}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    llm = LLM(model=str(args.model_root / "Qwen2-VL-7B-Instruct"),
              trust_remote_code=True, max_model_len=2048,
              limit_mm_per_prompt={"image": 2}, gpu_memory_utilization=0.5,
              tensor_parallel_size=1, dtype="bfloat16", enforce_eager=True)
    calls = []

    class Trace:
        def chat(self, **kwargs):
            outputs = llm.chat(**kwargs)
            assert outputs and outputs[0].outputs and outputs[0].outputs[0].text.strip()
            calls.append({"response": outputs[0].outputs[0].text,
                          "output_tokens": len(outputs[0].outputs[0].token_ids),
                          "input_tokens": len(outputs[0].prompt_token_ids)})
            return outputs

    trace = Trace()
    source = Image.open(args.image).convert("RGB")
    target = Image.open(args.target).convert("RGB")
    one_prompt = "What text do you see in this image? Output only the text content, nothing else."
    two_prompt = ('The first image is the original, and the second image reflects the changes made according '
                  'to the editing instruction in subject addition. Can you determine if the editing instruction '
                  'was successfully applied?\nThe editing instruction is: Add bench on the left of painted white leaf pattern\n\n'
                  'Please respond with "yes" or "no."')
    one = namespace["_query_vlm"](trace, source, one_prompt)
    two = namespace["_query_vlm_2_image"](trace, source, target, two_prompt)
    assert len(calls) == 2, "Upstream helper swallowed a failed VLM call"
    assert one and two
    return {"model": str(args.model_root / "Qwen2-VL-7B-Instruct"), "official_helpers": sorted(names),
            "sampling": {"temperature": 0.0, "max_tokens": 1024, "min_tokens": 1},
            "single_image_response": one, "two_image_response": two, "calls": calls,
            "note": "Component execution check; responses are not benchmark scores."}


def hps(args):
    import torch
    import yaml
    import hpsv3
    from hpsv3 import HPSv3RewardInferencer

    original = Path(hpsv3.__file__).parent / "config/HPSv3_7B.yaml"
    config = yaml.safe_load(original.read_text())
    config["model_name_or_path"] = str(args.model_root / "Qwen2-VL-7B-Instruct")
    config["output_dir"] = str(Path(os.environ["TMPDIR"]) / "hps-output")
    local_config = Path(os.environ["TMPDIR"]) / "HPSv3.local.yaml"
    local_config.write_text(yaml.safe_dump(config))
    infer = HPSv3RewardInferencer(config_path=str(local_config),
                                checkpoint_path=str(args.model_root / "HPSv3/HPSv3.safetensors"), device="cuda:0")
    with torch.inference_mode():
        rewards = infer.reward(prompts=["", ""], image_paths=[str(args.image), str(args.target)])
    assert rewards.shape == (2, 2) and torch.isfinite(rewards).all()
    return {"strict_state_load": True, "reward_shape": list(rewards.shape), "rewards": rewards.cpu().tolist(),
            "original_config_sha256": file_sha(original), "local_model_path": config["model_name_or_path"],
            "original_config": str(original)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["metrics", "judge", "hps"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-root", type=Path, default=Path("/home/chs/model"))
    ap.add_argument("--official-source", type=Path, default=Path("/home/chs/tools/EdiVal/96d34b00d7ea2dc3de90f2bb01f292f6dec6294a"))
    sample = Path("/home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/edival/full_512_bare_20261005_attention/run/edival/0")
    ap.add_argument("--image", type=Path, default=sample / "turn_0_input.png")
    ap.add_argument("--target", type=Path, default=sample / "turn_1.png")
    args = ap.parse_args()
    manifest = json.loads((args.official_source / "source.json").read_text())
    for name, sha in manifest["files"].items():
        assert file_sha(args.official_source / name) == sha, f"Official source changed: {name}"
    sys.path.insert(0, str(args.official_source))
    import torch
    assert torch.cuda.device_count() == 1, "Expose only physical GPU1"
    record = {"mode": args.mode, "started_at": time.time(), "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "official_revision": manifest["revision"], "official_files_verified": len(manifest["files"]),
              "image_sha256": file_sha(args.image), "target_sha256": file_sha(args.target),
              "torch": torch.__version__, "vllm_v2_runner": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
              "status": "running"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        record["results"] = {"metrics": metrics, "judge": judge, "hps": hps}[args.mode](args)
        record["status"] = "passed"
        record["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["ended_at"] = time.time()
        args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""GPU environment checks, not MICE capability scoring.

Stop the selected GPU's burn before running; restore it after the process exits.
Use mice-metrics for --mode metrics, mice-judge for --mode judge.
"""
import argparse
import base64
import gc
import io
import json
from pathlib import Path
import time


def metrics(root):
    import torch
    import torch.nn.functional as F
    import timm
    from PIL import Image, PngImagePlugin
    from groundingdino.models.GroundingDINO.ms_deform_attn import (
        MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.inference import load_image, predict
    PngImagePlugin.MAX_TEXT_CHUNK = 4 * 1024 * 1024
    torch.manual_seed(42)
    torch.set_num_threads(8)
    # Validate the modern Tensor API patch against upstream's PyTorch reference.
    value = torch.rand(1, 20, 2, 8, device="cuda")
    shapes = torch.tensor([[4, 4], [2, 2]], dtype=torch.long, device="cuda")
    starts = torch.tensor([0, 16], dtype=torch.long, device="cuda")
    locations = torch.rand(1, 5, 2, 2, 4, 2, device="cuda")
    weights = torch.rand(1, 5, 2, 2, 4, device="cuda")
    weights /= weights.sum(dim=(-1, -2), keepdim=True)
    native = MultiScaleDeformableAttnFunction.apply(value, shapes, starts, locations, weights, 1)
    reference = multi_scale_deformable_attn_pytorch(value, shapes, locations, weights)
    torch.testing.assert_close(native, reference, atol=2e-5, rtol=2e-5)
    error = (native - reference).abs().max().item()
    cfg = SLConfig.fromfile(str(root / "GroundingDINO-SwinT-OGC/GroundingDINO_SwinT_OGC.local.py"))
    cfg.device = "cuda"
    detector = build_model(cfg)
    checkpoint = torch.load(root / "GroundingDINO-SwinT-OGC/groundingdino_swint_ogc.pth",
                            map_location="cpu", weights_only=True)
    state = clean_state_dict(checkpoint["model"])
    incompatible = detector.load_state_dict(state, strict=False)
    # Official checkpoint can contain unused BERT position buffers and label encoder.
    if incompatible.missing_keys:
        raise ValueError(f"Missing detector weights: {incompatible.missing_keys}")
    allowed = {"bert.embeddings.position_ids", "bert.embeddings.token_type_ids", "label_enc.weight"}
    if set(incompatible.unexpected_keys) - allowed:
        raise ValueError(f"Unexpected detector weights: {incompatible.unexpected_keys}")
    detector = detector.eval().cuda()
    source = Path("/home/chs/dataset/MICE-Bench/source_image/091097244653b968.jpg")
    _, image = load_image(str(source))
    boxes, confidence, phrases = predict(detector, image, "sign . plate . person .", 0.3, 0.25)
    assert torch.isfinite(boxes).all() and torch.isfinite(confidence).all()
    if len(boxes) == 0:
        raise ValueError("Detector smoke image returned no objects")
    detection = {"boxes": boxes.tolist(), "confidence": confidence.tolist(), "phrases": phrases,
                 "missing_keys": incompatible.missing_keys, "unexpected_keys": incompatible.unexpected_keys}
    del detector, checkpoint, state
    gc.collect()
    torch.cuda.empty_cache()
    dino = timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=False,
                            checkpoint_path=str(root / "DINOv3-ViT-L-16/model.safetensors")).eval().cuda()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(dino), is_training=False)
    im = Image.open(source).convert("RGB")
    other = Image.new("RGB", im.size, "black")
    inputs = torch.stack([transform(im), transform(im), transform(other)]).cuda()
    with torch.inference_mode():
        features = dino.forward_features(inputs)
    assert torch.isfinite(features).all()
    prefix = dino.num_prefix_tokens
    patches = features[:, prefix:]
    assert patches.shape[1] == (inputs.shape[2] // 16) * (inputs.shape[3] // 16)
    cls = features[:, 0]
    sims = F.cosine_similarity(cls[0:1], cls[1:]).tolist()
    assert sims[0] > 0.9999 and sims[1] < sims[0] - 1e-4
    return {"extension_max_absolute_error": error, "detection": detection,
            "dino_features_shape": list(features.shape), "dino_prefix_tokens": prefix,
            "dino_patch_shape": list(patches.shape), "dino_cls_similarity_same_and_different": sims}


def judge(model):
    from PIL import Image
    from vllm import LLM, SamplingParams
    colors = ["red", "green", "blue", "yellow"]
    content = []
    for color in colors:
        data = io.BytesIO()
        Image.new("RGB", (256, 256), color).save(data, format="PNG")
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()}})
    content.append({"type": "text", "text":
        'List the dominant color of each image in order. Reply only with a JSON object: {"colors": [color names]}.'})
    llm = LLM(model=str(model), dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=0.88, max_model_len=8192, max_num_seqs=1,
              limit_mm_per_prompt={"image": 4}, enforce_eager=True,
              seed=42, disable_log_stats=True)
    output = llm.chat([{"role": "user", "content": content}],
                      sampling_params=SamplingParams(temperature=0, max_tokens=256, seed=42),
                      chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)[0]
    reply = output.outputs[0]
    if not reply.text.strip() or reply.finish_reason == "length":
        raise ValueError("Empty or truncated multimodal response")
    # Inspect semantic input handling, not just successful model loading.
    text = reply.text.strip()
    start, end = text.find("{"), text.rfind("}")
    parsed = json.loads(text[start:end + 1])
    if [c.lower().strip() for c in parsed["colors"]] != colors:
        raise ValueError("Four-image color order check failed")
    return {"model": str(model), "images": 4, "response": reply.text,
            "finish_reason": reply.finish_reason, "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(reply.token_ids), "dtype": "bfloat16",
            "max_model_len": 8192, "tensor_parallel_size": 1, "enforce_eager": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["metrics", "judge"], required=True)
    ap.add_argument("--model-root", type=Path, default=Path("/home/chs/model"))
    ap.add_argument("--model", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    started = time.time()
    result = {"status": "running", "mode": args.mode, "scope": "environment smoke only; not benchmark scores"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    try:
        result["details"] = metrics(args.model_root) if args.mode == "metrics" else judge(args.model)
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["elapsed_seconds"] = time.time() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

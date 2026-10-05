"""Explicit GPU acceptance probe; never run as part of installation."""

import argparse
import copy
import os
from pathlib import Path

from PIL import Image

from lance_mice.backend import OmniBackend, find_history_metadata, history_payload
from lance_mice.dataset import load_samples
from lance_mice.images import bucket_size, image_hash, prepare_vae_image
from lance_mice.protocol import derive_seed
from lance_mice.runner import write_json
from lance_mice.settings import Settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("/home/chs/model/Lance"))
    p.add_argument("--dataset", type=Path, default=Path("/home/chs/dataset/MICE-Bench"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", default="0")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.output.mkdir(parents=True, exist_ok=False)
    sample = load_samples(args.dataset)[0]
    settings = Settings()
    with Image.open(sample.image) as image:
        image = prepare_vae_image(image, bucket_size(*image.size))
    image.save(args.output / "common_input.png")
    payload = history_payload(sample.session_id, [image], [sample.instructions[0]], settings, audit=True)
    history = payload["extra_args"].pop("lance_history")
    history["probe_output"] = str(args.output.resolve())
    payload["extra_args"]["single_turn_probe"] = history
    payload["extra_args"].update({k: getattr(settings, k) for k in
        ("timestep_shift", "cfg_text_scale", "cfg_img_scale", "cfg_interval", "cfg_renorm_type", "cfg_renorm_min")})
    backend = OmniBackend(args.model, settings,
        pipeline_class="lance_mice.single_turn_probe.SingleTurnProbePipeline")
    try:
        params = copy.deepcopy(backend.engine.default_sampling_params_list)
        params[0].num_inference_steps = settings.steps
        params[0].seed = derive_seed(settings.seed, sample.session_id, "noise", 1)
        outputs = list(backend.engine.generate(payload, sampling_params_list=params, use_tqdm=False))
        if len(outputs) != 1 or len(outputs[0].images or []) != 1:
            raise RuntimeError("Expected one probe output")
        outputs[0].images[0].save(args.output / "adapter.png")
        report = find_history_metadata(outputs[0], "single_turn_probe")
        report.update({"session_id": sample.session_id, "instruction": sample.instructions[0],
                       "settings": settings.identity(), "input_hash": image_hash(image)})
        write_json(args.output / "comparison.json", report)
        print(report)
        return 0 if report["status"] == "passed" else 1
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())

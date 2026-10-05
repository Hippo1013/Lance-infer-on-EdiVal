"""GPU-only diagnostic: official and custom single-turn numerical parity.

This pipeline is selected only by the explicit probe command. It omits the
production history framing and uses one common encoding seed so that both
paths receive identical conditions. Production prompting/RNG are unchanged.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.lance.pipeline_lance import LancePipeline

from .omni_pipeline import LanceHistoryPipeline
from .protocol import Segment


def snapshot(kwargs):
    tensors, values = {}, {}
    for name, value in kwargs.items():
        if torch.is_tensor(value):
            tensors[name] = value.detach().cpu().clone()
        elif name.endswith("past_key_values") and value is not None:
            for kind in ("key_cache", "value_cache"):
                for index, tensor in getattr(value, kind).items():
                    if tensor is not None:
                        tensors[f"{name}.{kind}.{index}"] = tensor.detach().cpu().clone()
            values[f"{name}.lens"] = value.key_values_lens
        else:
            values[name] = value
    return tensors, values


class SingleTurnProbePipeline(LanceHistoryPipeline):
    def _history_segments(self, instructions):
        return [Segment("image", "image", 0, image_index=0),
                Segment("text", "label", 1, ""),
                Segment("text", "current", 1, instructions[0])]

    @contextmanager
    def _rng(self, seed):
        with super()._rng(self._probe_seed):
            yield

    @torch.inference_mode()
    def forward(self, req):
        prompt = req.prompts[0]
        history = (prompt.get("extra_args") or {}).get("single_turn_probe") if isinstance(prompt, dict) else None
        if history is None:
            # Omni's startup dummy request is not a parity probe.
            return super().forward(req)
        if len(history["instructions"]) != 1:
            raise ValueError("Probe requires exactly one input image/instruction")
        self._probe_seed = req.sampling_params.seed
        original = self.bagel.generate_image
        captures = []

        def capture(**kwargs):
            inputs = snapshot(kwargs)
            result = original(**kwargs)
            captures.append((inputs, result[0][0].detach().cpu().clone()))
            return result

        self.bagel.generate_image = capture
        try:
            official = LancePipeline.forward(self, req)
            candidate = self._forward_history(req, history)
        finally:
            self.bagel.generate_image = original
            self._clear_history()
        (a, av), al = captures[0]
        (b, bv), bl = captures[1]
        differences = []
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b or a[key].shape != b[key].shape:
                differences.append({"tensor": key, "reason": "missing or shape mismatch"})
            elif not torch.equal(a[key], b[key]):
                differences.append({"tensor": key,
                    "max_absolute_error": float((a[key].float() - b[key].float()).abs().max())})
        left = np.asarray(official.output["payload"]["image"].convert("RGB"))
        right = np.asarray(candidate.output["payload"]["image"].convert("RGB"))
        pixel_equal = left.shape == right.shape and np.array_equal(left, right)
        report = {"scope": "Same image, raw instruction, encoding seed, noise and denoising settings; no history framing",
                  "input_tensor_count": len(a), "input_differences": differences,
                  "scalar_inputs_equal": av == bv,
                  "latents_equal": torch.equal(al, bl),
                  "latent_max_absolute_error": float((al.float() - bl.float()).abs().max()),
                  "identical_pixels": pixel_equal,
                  "pixel_mae_0_255": float(np.abs(left.astype(float)-right.astype(float)).mean())
                      if left.shape == right.shape else None}
        report["status"] = "passed" if not differences and av == bv and torch.equal(al, bl) and pixel_equal else "failed"
        official.output["payload"]["image"].save(Path(history["probe_output"]) / "official.png")
        return DiffusionOutput(output={"payload": candidate.output["payload"],
            "metadata": {"image": candidate.output["metadata"]["image"], "single_turn_probe": report}})

"""Offline Omni bridge; importing this module does not import CUDA libraries."""

from __future__ import annotations

import copy
import os
from dataclasses import asdict
from pathlib import Path

from .images import image_hash
from .protocol import PROTOCOL_VERSION, derive_seed
from .settings import Settings


def history_payload(session_id, images, instructions, settings, *, audit=False, end_session=False, attention_path=None):
    if not instructions or len(images) != len(instructions):
        raise ValueError("Images and instructions must have equal nonzero length")
    payload = {
        # Structured history is authoritative; the stock text extractor is bypassed.
        "prompt": instructions[-1], "modalities": ["image"],
        "multi_modal_data": {"image": images},
        "extra_args": {"lance_history": {
            "protocol": PROTOCOL_VERSION, "session_id": session_id,
            "instructions": list(instructions), "image_hashes": [image_hash(im) for im in images],
            "settings": asdict(settings), "audit": audit, "end_session": end_session,
        }},
    }
    if attention_path is not None:
        payload["extra_args"]["lance_history"]["attention_output"] = str(Path(attention_path).resolve())
    return payload


def find_history_metadata(output, name="lance_history"):
    def search(obj, depth=0):
        if not isinstance(obj, dict) or depth > 5:
            return None
        if isinstance(obj.get(name), dict):
            return obj[name]
        for key in ("metadata", "metrics", "custom_output", "output", "multimodal_output"):
            found = search(obj.get(key), depth + 1)
            if found is not None:
                return found
        return None
    for key in ("multimodal_output", "metrics", "output"):
        found = search(getattr(output, key, None))
        if found is not None:
            return found
    raise RuntimeError(f"Omni returned an image without the required {name} audit metadata")


class OmniBackend:
    def __init__(self, model: Path, settings: Settings, *, cfg_parallel_size=1, audit=False,
                 pipeline_class="lance_mice.omni_pipeline.LanceHistoryPipeline"):
        if cfg_parallel_size not in (1, 2):
            raise ValueError("Only one or two CFG ranks are supported")
        if cfg_parallel_size == 2 and settings.cfg_text_scale <= 1:
            raise ValueError("CFG-2 requires text guidance > 1")
        self.settings, self.audit = settings, audit
        self.generated_turns = []
        # Must be set before vLLM/Torch imports and before spawning workers.
        os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "FLASH_ATTN")
        from vllm_omni.entrypoints.omni import Omni

        self.engine = Omni(
            model=str(model.resolve()), dtype="bfloat16",
            num_gpus=cfg_parallel_size, parallel_config={"cfg_parallel_size": cfg_parallel_size},
            max_num_seqs=1, max_num_batched_tokens=32768,
            enforce_eager=True, step_execution=False, enable_prefix_caching=False,
            trust_remote_code=True, async_chunk=False,
            diffusion_load_format="custom_pipeline",
            custom_pipeline_args={"pipeline_class": pipeline_class},
        )

    def edit(self, session_id, images, instructions, *, end_session=False, attention_path=None):
        payload = history_payload(session_id, images, instructions, self.settings,
                                  audit=self.audit, end_session=end_session, attention_path=attention_path)
        params = copy.deepcopy(self.engine.default_sampling_params_list)
        params[0].num_inference_steps = self.settings.steps
        params[0].seed = derive_seed(self.settings.seed, session_id, "noise", len(instructions))
        outputs = list(self.engine.generate(payload, sampling_params_list=params, use_tqdm=False))
        if len(outputs) != 1 or len(outputs[0].images or []) != 1:
            raise RuntimeError("Expected exactly one output image")
        self.generated_turns.append([session_id, len(instructions)])
        return outputs[0].images[0].convert("RGB"), find_history_metadata(outputs[0])

    def close(self):
        self.engine.close()

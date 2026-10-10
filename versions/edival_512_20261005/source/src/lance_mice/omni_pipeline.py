# SPDX-License-Identifier: Apache-2.0
# Portions adapted from vLLM-Omni's Lance/BAGEL pipelines, Copyright contributors
# to the vLLM-Omni project. See THIRD_PARTY_NOTICES.md for pinned sources.
"""Full-history Lance pipeline. Engineering acceptance includes full-history input and cache equivalence.

Do not enable the inherited BAGEL step scheduler: it bypasses this forward.
Only immutable context before the current label may persist across requests.
"""

from __future__ import annotations

import copy
import time
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from PIL import Image
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.lance.lance_transformer import NaiveCache
from vllm_omni.diffusion.models.lance.pipeline_lance import LancePipeline

from .images import bucket_size, image_hash, prepare_vae_image
from .protocol import (PROTOCOL_VERSION, derive_seed, digest, history_segments,
                       prefix_length, reusable_prefix, segment_signature)
from .settings import Settings


def clone_context(ctx: dict) -> dict:
    """Deep-copy writable cache tensors; never mutate a saved prefix or sibling."""
    result = {k: copy.deepcopy(v) for k, v in ctx.items() if k != "past_key_values"}
    source = ctx["past_key_values"]
    cache = NaiveCache(source.num_layers)
    for name in ("key_cache", "value_cache"):
        setattr(cache, name, {k: None if v is None else v.clone()
                             for k, v in getattr(source, name).items()})
    cache.key_values_lens = copy.deepcopy(source.key_values_lens)
    result["past_key_values"] = cache
    return result


class LanceHistoryPipeline(LancePipeline):
    supports_step_execution = False

    def __init__(self, *, od_config, prefix: str = ""):
        if getattr(od_config, "step_execution", False):
            raise ValueError("LanceHistoryPipeline requires step_execution=False")
        super().__init__(od_config=od_config, prefix=prefix)
        # Lance exposes the same modules both at the pipeline root and under
        # bagel. The custom loader's tied-weight dedup keeps the first name,
        # but Lance weights_sources uses bagel.*. Retain callable aliases
        # without registering duplicate parameter namespaces.
        for name in ("language_model", "transformer", "vit_model"):
            module = self._modules.pop(name, None)
            if module is not None:
                object.__setattr__(self, name, module)
        self.eval()
        self._clear_history()

    def _clear_history(self):
        self._session_key = None
        self._encoded = {}
        self._prefix = None
        self._prefix_signatures = []
        self._prefix_trace = []

    @contextmanager
    def _rng(self, seed):
        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with torch.random.fork_rng(devices=[device_index]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            yield

    @contextmanager
    def _measure(self, name):
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        yield
        torch.cuda.synchronize(self.device)
        self._times[name] = self._times.get(name, 0.0) + time.perf_counter() - start

    def _to_device(self, values):
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in values.items()}

    @torch.inference_mode()
    def forward(self, req):
        if len(req.prompts) != 1:
            raise ValueError("History pipeline accepts exactly one request per worker")
        prompt = req.prompts[0]
        if not isinstance(prompt, dict):
            raise ValueError("History pipeline requires a structured prompt")
        history = (prompt.get("extra_args") or {}).get("lance_history")
        if history is None:
            # Explicitly supported for the upstream single-turn reference test.
            self._clear_history()
            return super().forward(req)
        try:
            return self._forward_history(req, history)
        except BaseException:
            self._clear_history()
            raise

    def _prefill_image(self, ctx, image, image_index, session_id, settings, target_size, trace):
        key = digest([image_index, image_hash(image), target_size])
        entry = self._encoded.setdefault(key, {}) if settings.cache_mode != "none" else {}
        start = ctx["kv_lens"][0]
        rope_start = ctx["ropes"][0]
        if "vit_anchor" not in ctx:
            ctx["vit_anchor"] = rope_start

        # The upstream helper shifts EVERY call's ViT block to t=1000. For a
        # history, preserve relative offsets using the first image's anchor.
        with self._rng(derive_seed(settings.seed, session_id, "vit", image_index)):
            inp, lens, ropes = self.bagel.prepare_vit_images(
                curr_kvlens=ctx["kv_lens"], curr_rope=ctx["ropes"], images=[image],
                transforms=None, new_token_ids=self.new_token_ids)
        inp = self._to_device(inp)
        inp["packed_position_ids"][0] += rope_start - ctx["vit_anchor"]

        if "vit" not in entry:
            with self._measure("vit_encode"), torch.autocast(**self._autocast_kwargs()):
                self.bagel.vit_model.set_pending_grid_thw(inp["_lance_grid_thw"])
                cu = torch.nn.functional.pad(torch.cumsum(inp["vit_token_seqlens"], 0), (1, 0)).int()
                embed = self.bagel.vit_model(
                    packed_pixel_values=inp["packed_vit_tokens"],
                    packed_flattened_position_ids=inp["packed_vit_position_ids"],
                    cu_seqlens=cu, max_seqlen=int(inp["vit_token_seqlens"].max()))
                entry["vit"] = (self.bagel.connector(embed)
                                + self.bagel.vit_pos_embed(inp["packed_vit_position_ids"]))
                self._counts["vit_encodes"] += 1

        with torch.autocast(**self._autocast_kwargs()):
            text_embed = self.bagel.language_model(
                packed_text_ids=inp["packed_text_ids"], return_embeddings_only=True).packed_query_sequence
            sequence = text_embed.new_zeros((int(inp["packed_seqlens"].sum()), self.bagel.hidden_size))
            sequence[inp["packed_text_indexes"]] = text_embed
            sequence[inp["packed_vit_token_indexes"]] = entry["vit"].to(sequence.dtype)
            kwargs = {"mode": "und"} if self.bagel.use_moe else {}
            ctx["past_key_values"] = self.bagel.language_model(
                packed_query_sequence=sequence, query_lens=inp["packed_seqlens"],
                packed_query_position_ids=inp["packed_position_ids"],
                past_key_values=ctx["past_key_values"], update_past_key_values=True,
                is_causal=False, **kwargs).past_key_values
        ctx["kv_lens"], ctx["ropes"] = lens, ropes
        trace.append({"kind": "vit", "image_index": image_index, "tokens": [start, lens[0]],
                      "position_hash": self._position_hash(inp["packed_position_ids"])})

        # Each image has its own VAE sample. Both CFG branches later fork this
        # shared prefix, so they cannot accidentally receive different samples.
        vae_image = prepare_vae_image(image, target_size)
        def vae_transform(im):
            return torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1) / 127.5 - 1

        start = ctx["kv_lens"][0]
        ctx["latest_vae_rope"] = ctx["ropes"][0]
        inp, lens, ropes = self.bagel._lance_native_prepare_vae_images(
            curr_kvlens=ctx["kv_lens"], curr_rope=ctx["ropes"], images=[vae_image],
            transforms=vae_transform, new_token_ids=self.new_token_ids, is_video=False)
        inp = self._to_device(inp)
        if "vae" not in entry:
            with self._measure("vae_encode"), self._rng(derive_seed(settings.seed, session_id, "vae", image_index)):
                entry["vae"] = self.vae.encode(inp["padded_images"])
                self._counts["vae_encodes"] += 1
        inp["precomputed_latent"] = entry["vae"]
        with torch.autocast(**self._autocast_kwargs()):
            ctx["past_key_values"] = self.bagel.forward_cache_update_vae(
                self.vae, ctx["past_key_values"], **inp)
        ctx["kv_lens"], ctx["ropes"] = lens, ropes
        trace.append({"kind": "vae", "image_index": image_index, "tokens": [start, lens[0]],
                      "position_hash": self._position_hash(inp["packed_position_ids"])})

    def _position_hash(self, tensor):
        return digest(tensor.tolist()) if self._audit else None

    def _history_segments(self, instructions):
        return history_segments(instructions)

    def _text(self, ctx, segment, trace):
        start = ctx["kv_lens"][0]
        self._raw_text_prefill(ctx, segment.text)
        trace.append({"kind": "text", "role": segment.role, "turn": segment.turn,
                      "tokens": [start, ctx["kv_lens"][0]], "text": segment.text})

    def _forward_history(self, req, history):
        if history.get("protocol") != PROTOCOL_VERSION:
            raise ValueError("History protocol version mismatch")
        settings = Settings(**{**history["settings"], "cfg_interval": tuple(history["settings"]["cfg_interval"])})
        instructions = history["instructions"]
        session_id = history["session_id"]
        images = (req.prompts[0].get("multi_modal_data") or {}).get("image")
        if not isinstance(images, list) or len(images) != len(instructions):
            raise ValueError("Need one ordered input image for every instruction")
        if not all(isinstance(im, Image.Image) for im in images):
            raise ValueError("History images must be decoded PIL images")
        segments = self._history_segments(instructions)
        image_hashes = [image_hash(im) for im in images]
        if image_hashes != history["image_hashes"]:
            raise ValueError("Image content changed in request transport")
        seed = derive_seed(settings.seed, session_id, "noise", len(instructions))
        if req.sampling_params.seed != seed or req.sampling_params.num_inference_steps != settings.steps:
            raise ValueError("Sampling parameters disagree with history settings")

        session_key = digest([session_id, settings.identity(), history.get("audit", False)])
        if session_key != self._session_key or len(instructions) == 1:
            self._clear_history()
        self._session_key = session_key
        self._audit = bool(history.get("audit", False))
        self._times, self._counts = {}, {"vit_encodes": 0, "vae_encodes": 0}
        torch.cuda.reset_peak_memory_stats(self.device)
        target_size = bucket_size(*images[0].size, settings.resolution)
        # Flat learned positional indices must stay in the released table.
        w, h = target_size
        if (h // 16 - 1) * self.bagel.max_latent_size + w // 16 - 1 >= self.bagel.max_latent_size**2:
            raise ValueError(f"Target bucket exceeds learned positional table: {target_size}")

        boundary = prefix_length(segments)
        signatures = [segment_signature(s, image_hashes) for s in segments[:boundary]]
        reused = (reusable_prefix(self._prefix_signatures, signatures)
                  if settings.cache_mode == "prefix" and self._prefix is not None else 0)
        header, suffix = self._segment_strings("image_edit", "image")
        with self._measure("prefill_total"):
            if reused:
                ctx, trace = clone_context(self._prefix), copy.deepcopy(self._prefix_trace)
            else:
                ctx, trace = self._new_gen_context(), []
                self._raw_text_prefill(ctx, header)
            for segment in segments[reused:boundary]:
                if segment.kind == "image":
                    self._prefill_image(ctx, images[segment.image_index], segment.image_index,
                                        session_id, settings, target_size, trace)
                else:
                    self._text(ctx, segment, trace)
            if settings.cache_mode == "prefix":
                self._prefix = clone_context(ctx)
                self._prefix_signatures = signatures
                self._prefix_trace = copy.deepcopy(trace)

            gen = clone_context(ctx)
            neg = clone_context(ctx) if settings.cfg_text_scale > 1 else None
            gen_trace = copy.deepcopy(trace)
            neg_trace = copy.deepcopy(trace) if neg is not None else []
            for segment in segments[boundary:]:
                self._text(gen, segment, gen_trace)
                if neg is not None and segment.role != "current":
                    self._text(neg, segment, neg_trace)
            self._raw_text_prefill(gen, suffix)
            if neg is not None:
                self._raw_text_prefill(neg, suffix)

        shape = (h, w)
        gen_input = self.bagel.prepare_vae_latent(
            curr_kvlens=gen["kv_lens"], curr_rope=[ctx["latest_vae_rope"]],
            image_sizes=[shape], new_token_ids=self.new_token_ids)
        gen_input = self._to_device(gen_input)
        neg_input = self.bagel.prepare_vae_latent_cfg(
            curr_kvlens=(neg or gen)["kv_lens"], curr_rope=[ctx["latest_vae_rope"]], image_sizes=[shape])
        neg_input = self._to_device(neg_input)
        self._regen_init_noise_on_device(gen_input, seed)
        for key in ("key_values_lens", "packed_indexes", "packed_key_value_indexes"):
            gen_input.pop(key, None)

        observer = None
        if history.get("attention_output"):
            from .attention import AttentionObserver
            observer = AttentionObserver(self.bagel, gen_trace, gen["kv_lens"][0],
                int(gen_input["packed_text_ids"].numel()), int(gen_input["packed_init_noises"].shape[0]), settings.steps)
        with self._measure("denoise"), torch.autocast(**self._autocast_kwargs()), (observer or nullcontext()):
            latents, *_ = self.bagel.generate_image(
                past_key_values=gen["past_key_values"],
                cfg_text_past_key_values=(neg or gen)["past_key_values"],
                cfg_img_past_key_values=gen["past_key_values"],
                num_timesteps=settings.steps, timestep_shift=settings.timestep_shift,
                cfg_text_scale=settings.cfg_text_scale, cfg_img_scale=1.0,
                cfg_interval=settings.cfg_interval, cfg_renorm_type=settings.cfg_renorm_type,
                cfg_renorm_min=settings.cfg_renorm_min,
                cfg_text_packed_position_ids=neg_input["cfg_packed_position_ids"],
                cfg_img_packed_position_ids=None, **gen_input)
        with self._measure("decode"):
            image = self._decode_image_from_latent(self.bagel, self.vae, latents[0], shape)

        self._times["prefill_excluding_encoders"] = max(0.0, self._times["prefill_total"]
            - self._times.get("vit_encode", 0) - self._times.get("vae_encode", 0))
        metadata = {
            "protocol": PROTOCOL_VERSION, "session_id": session_id, "turn": len(instructions),
            "image_hashes": image_hashes, "seed": seed, "size": target_size,
            "prefix_segments_reused": reused, "counts": self._counts,
            "seconds": self._times, "positive_trace": gen_trace, "negative_trace": neg_trace,
            "positive_kv_tokens": gen["kv_lens"][0],
            "negative_kv_tokens": (neg or gen)["kv_lens"][0],
            "peak_memory_bytes": torch.cuda.max_memory_allocated(self.device),
        }
        if observer is not None:
            metadata["attention"] = observer.save(history["attention_output"])
        if history.get("end_session") or settings.cache_mode == "none":
            self._clear_history()
        return DiffusionOutput(output={"payload": {"image": image},
                                       "metadata": {"image": {"shape": shape}, "lance_history": metadata}},
                               stage_durations=dict(self._times))

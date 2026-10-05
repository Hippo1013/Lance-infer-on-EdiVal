# Third-party code

`src/lance_mice/omni_pipeline.py` adapts the segmented prefill, ViT cache update,
and denoising calls from the following Apache-2.0 sources:

- vLLM-Omni, copyright contributors to the vLLM-Omni project, commit
  `b742f86136d28423903a1213adde2a62a11067a2`:
  `vllm_omni/diffusion/models/lance/pipeline_lance.py`,
  `vllm_omni/diffusion/models/lance/lance_transformer.py`, and
  `vllm_omni/diffusion/models/bagel/bagel_transformer.py`.
  <https://github.com/vllm-project/vllm-omni>
- The bucket geometry in `src/lance_mice/images.py` is adapted from Lance,
  copyright (c) 2025 ByteDance Ltd. and/or its affiliates, commit
  `4baeee086648996f6ab12e673cbe461b0b149997`,
  `data/video/transforms/bucket_resize.py`.
  <https://github.com/bytedance/Lance>

The Apache-2.0 license for these portions is reproduced in
`LICENSES/Apache-2.0.txt`. Changes include history segment compilation, explicit
current-instruction CFG masking, per-image RNG, session-local feature/KV caches,
global ViT temporal offsets, and benchmark audit metadata.

No model weights or dataset assets are included in this repository. Their
respective upstream terms continue to apply.

## Edit-R2 scoring reference

`src/lance_mice/scoring/if_rules.py` and `prompts.py` adapt selected IF rules and retain GA prompt strings from Edit-R2 commit `26b55829246e1a67fc3c8d522324fee3d49cd954`. See `src/lance_mice/scoring/NOTICE` for exact files, hashes and modification scope. No LICENSE was found in that upstream revision; the Apache-2.0 statement above does not apply to these portions. The original provenance and license scope are retained; no additional license is asserted for these upstream portions.

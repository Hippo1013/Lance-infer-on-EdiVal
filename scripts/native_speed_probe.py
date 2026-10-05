#!/usr/bin/env python3
"""Small official Lance speed probe with the production history protocol.

Run in a compatible native Lance environment, not the Omni environment. Source
checkout and its dependencies are supplied externally; this script never installs
or modifies them. This is a timing trial, not cross-framework numerical parity.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lance_mice.dataset import load_samples
from lance_mice.images import image_hash
from lance_mice.protocol import derive_seed, history_segments, protocol_manifest
from lance_mice.runner import write_json

NATIVE_REVISION = '4baeee086648996f6ab12e673cbe461b0b149997'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-source', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('/home/chs/model/Lance'))
    p.add_argument('--dataset', type=Path, default=Path('/home/chs/dataset/MICE-Bench'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.model = args.model.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Probe output must be empty')
    samples = load_samples(args.dataset)
    plan, rows = [], []
    for phase in ('warmup', 'timed'):
        for sample in samples:
            paths = [sample.image]
            for turn in range(1, len(sample.instructions) + 1):
                index = len(rows)
                segments = history_segments(list(sample.instructions[:turn]))
                elements = [str(paths[s.image_index]) if s.kind == 'image' else s.text for s in segments]
                rows.append({'index': index, 'data': {'interleave_array': elements,
                    'element_dtype_array': [s.kind for s in segments],
                    'istarget_in_interleave': [0] * len(segments)}})
                plan.append({'index': index, 'phase': phase, 'session': sample.session_id,
                    'turn': turn, 'input_paths': list(map(str, paths)),
                    'noise_seed': derive_seed(42, sample.session_id, 'noise', turn),
                    'protocol': protocol_manifest(list(sample.instructions[:turn]))})
                paths.append(args.output / f'{index:06d}.png')
    config = args.output / 'native_inputs.json'
    write_json(config, rows)
    write_json(args.output / 'plan.json', plan)
    sys.path.insert(0, str(args.native_source.resolve()))
    os.chdir(args.native_source)
    import torch
    from PIL import Image
    import inference_lance as native
    import data.datasets_custom.validation_dataset as vd

    # Native chooses vision_type from the final interleaved element (our current
    # instruction is text). Keep its official IMAGE-edit system template.
    original_system = vd.generate_system_prompt
    def system_prompt(system_prompt_type="caption", vision_type="video"):
        return original_system(system_prompt_type, "image" if system_prompt_type == "image_edit" else vision_type)
    vd.generate_system_prompt = system_prompt
    write_json(args.output / 'system_prompt.json', {'text': system_prompt('image_edit', 'image')})

    # The stock helper selects the first text match. Select the last match so
    # repeated historical instructions never become the current CFG drop span.
    original_expand = vd.expand_and_index_by_token_ids_new
    spans = []
    def expand(*a, **kw):
        target = kw.pop('search_text', '')
        ids, blocks, target_ids, _ = original_expand(*a, search_text='', **kw)
        needle = kw['tokenizer'](target, add_special_tokens=False)['input_ids']
        hits = [i for i in range(len(ids) - len(needle) + 1) if ids[i:i+len(needle)] == needle]
        if not needle or not hits:
            raise ValueError('Current instruction not found in native token stream')
        start = hits[-1]
        span = list(range(start, start + len(needle)))
        if span[-1] >= target_ids[0]:
            raise ValueError('Current instruction overlaps generation target')
        spans.append({'current_cfg_span': [start, start + len(needle)],
                      'current_text': target, 'image_blocks': len(blocks), 'tokens': len(ids)})
        return ids, blocks, target_ids, span
    vd.expand_and_index_by_token_ids_new = expand

    weight_report = {}
    original_load = native.init_from_model_path_if_needed
    def load(*a, **kw):
        result = original_load(*a, **kw)
        weight_report.update(missing=list(result.missing_keys), unexpected=list(result.unexpected_keys))
        write_json(args.output / 'weight_load.json', weight_report)
        # ViT was already loaded strictly from its separate checkpoint by native.main.
        allowed = {'latent_pos_embed.pos_embed'} | {'vit_model.' + k for k in a[0].vit_model.state_dict()}
        if set(result.missing_keys) - allowed or result.unexpected_keys:
            raise ValueError(f'Unexpected weight mismatch: {weight_report}')
        return result
    native.init_from_model_path_if_needed = load

    timing, active = [], {}
    original_item = vd.ValidationDataset.__getitem__
    def getitem(self, idx):
        row = plan[idx]
        torch.cuda.synchronize()
        active.update(start=time.perf_counter(), row=row)
        native.set_seed(derive_seed(42, row['session'], 'native_encoding', row['turn']) % (2**32))
        active['input_hashes'] = []
        for path in row['input_paths']:
            with Image.open(path) as im:
                active['input_hashes'].append(image_hash(im.convert('RGB')))
        return original_item(self, idx)
    vd.ValidationDataset.__getitem__ = getitem
    original_validate = native.validate_on_fixed_batch
    def validate(*a, **kw):
        row = active['row']
        kw['training_args'].validation_noise_seed = row['noise_seed']
        torch.cuda.reset_peak_memory_stats()
        original_validate(*a, **kw)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - active['start']
        with Image.open(args.output / f"{row['index']:06d}.png") as im:
            output_hash, size = image_hash(im.convert('RGB')), im.size
        timing.append({**row, **spans[-1], 'input_hashes': active['input_hashes'],
            'output_hash': output_hash, 'size': size, 'wall_seconds': elapsed,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
        write_json(args.output / 'turns.json', timing)
        print(f"NATIVE_PROBE {row['phase']} {row['session']} turn={row['turn']} seconds={elapsed:.3f}", flush=True)
    native.validate_on_fixed_batch = validate
    native_args = {'model_path': str(args.model / 'Lance_3B'), 'vit_path': str(args.model / 'Qwen2.5-VL-ViT'),
        'vit_type': 'qwen_2_5_vl_original', 'llm_qk_norm': 'true', 'llm_qk_norm_und': 'true',
        'llm_qk_norm_gen': 'true', 'tie_word_embeddings': 'false', 'validation_num_timesteps': '30',
        'validation_timestep_shift': '3.5', 'copy_init_moe': 'true', 'max_num_frames': '121',
        'max_latent_size': '64', 'visual_und': 'true', 'visual_gen': 'true', 'vae_model_type': 'wan',
        'apply_qwen_2_5_vl_pos_emb': 'true', 'apply_chat_template': 'false', 'cfg_type': '0',
        'validation_data_seed': '42', 'task': 'image_edit', 'save_path_gen': str(args.output),
        'resolution': 'image_768res', 'text_template': 'true', 'cfg_text_scale': '4',
        'use_KVcache': 'true', 'enhance_prompt': 'false', 'val_dataset_config_file': str(config)}
    sys.argv = ['inference_lance.py'] + [x for k, v in native_args.items() for x in (f'--{k}', v)] + ['--latent_patch_size', '1', '1', '1']
    write_json(args.output / 'runtime.json', {'native_revision': NATIVE_REVISION,
        'python': sys.version, 'packages': {name: importlib.metadata.version(name) for name in
        ('torch', 'torchvision', 'transformers', 'diffusers', 'flash-attn')}, 'arguments': sys.argv,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
    native.main()
    timed = [r for r in timing if r['phase'] == 'timed']
    # Verify each session uses precisely its own saved previous outputs.
    for i, r in enumerate(timing):
        if r['turn'] > 1 and r['input_hashes'][-1] != timing[i-1]['output_hash']:
            raise ValueError('Broken feedback chain')
    write_json(args.output / 'summary.json', {'status': 'completed', 'warmup_turns': 6,
        'timed_turns': len(timed), 'timed_sum_seconds': sum(r['wall_seconds'] for r in timed),
        'mean_seconds_per_turn': statistics.mean(r['wall_seconds'] for r in timed),
        'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in timing),
        'scope': 'One warmup and one timed CM/CU batch; native preprocessing, encoding, denoising, decode, save and in-call cleanup included. Model loading and outer-loop cleanup excluded. Native KV cache is within each turn; no cross-turn prefix reuse. Same prompt logic and sampling settings; native preprocessing/RNG/kernels differ, so this is not numerical parity or a controlled framework-only speed comparison.'})

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Verify bare full-history prompts with real cache/attention parity on both benchmarks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from lance_mice.acceptance import compare_runs, validate_run
from lance_mice.dataset import DEFAULT_ROOTS, load_samples, shard_samples
from lance_mice.protocol import PROTOCOL_VERSION
from lance_mice.runner import write_json

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    paths = list((ROOT/'src/lance_mice').glob('*.py')) + [ROOT/'pyproject.toml']
    paths += [ROOT/'scripts'/name for name in ('bare_acceptance.py', 'full_mice_job.py',
              'full_imgedit_job.py', 'validate_mice_full.py')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def check_workers(run, samples):
    pids = []
    for i in range(2):
        worker = json.loads((run/f'worker_{i}.json').read_text())
        expected = [[s.session_id, t] for s in shard_samples(samples, i, 2)
                    for t in range(1, len(s.instructions)+1)]
        if not worker['completed'] or worker['generated_turns'] != expected or worker['gpu'] != str(i):
            raise ValueError(f'Worker assignment differs: {run}/{i}')
        pids.append(worker['pid'])
    if len(set(pids)) != 2:
        raise ValueError('Expected two worker processes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pins = source_hashes()
    write_json(output/'source_hashes.json', pins)
    comparisons, counts = {}, {}
    try:
        for benchmark in ('mice', 'imgedit'):
            samples = load_samples(DEFAULT_ROOTS[benchmark], 'smoke', benchmark)
            for mode in ('none', 'prefix_attention'):
                run = output/benchmark/mode
                write_json(output/'status.json', {'state':'running', 'benchmark':benchmark, 'mode':mode})
                cmd = [sys.executable, '-m', 'lance_mice.runner', '--benchmark', benchmark,
                       '--profile', 'dp2', '--gpus', '0,1', '--audit', '--cache-mode',
                       'none' if mode == 'none' else 'prefix', '--output', str(run)]
                if mode == 'prefix_attention':
                    cmd.append('--save-attention')
                subprocess.run(cmd, cwd=ROOT, check=True)
                rows = validate_run(run)
                check_workers(run, samples)
                for key, row in rows.items():
                    traces = row['backend']['positive_trace']
                    if any(s.get('role') == 'framing' or
                           (s.get('role') == 'label' and (s['text'] or s['tokens'][0] != s['tokens'][1]))
                           for s in traces):
                        raise ValueError(f'Extra prompt text reached model: {key}')
                    if mode == 'prefix_attention' and row['turn'] > 1 and not row['backend']['prefix_segments_reused']:
                        raise ValueError(f'Prefix was not reused: {key}')
            comparison = compare_runs(output/benchmark/'none', output/benchmark/'prefix_attention')
            if not comparison['identical_pixels']:
                raise ValueError(f'Cache/attention changed pixels: {benchmark}')
            comparisons[benchmark] = comparison
            counts[benchmark] = comparison['turns']
            write_json(output/f'{benchmark}_summary.json', {'status':'passed', 'protocol':PROTOCOL_VERSION,
                'sessions':len(samples), 'turns':comparison['turns'], 'comparison':comparison})
        # Directly match the prior bare experiment, including its unchanged RNG.
        old = ROOT/'outputs/diagnostics/prompt_layouts_20261004/development/cm/05447e326032ca20/bare'
        new = output/'mice/prefix_attention/cm/05447e326032ca20'
        matched = []
        from PIL import Image
        from lance_mice.images import image_hash
        for turn in range(1, 4):
            with Image.open(old/f'turn_{turn}.png') as a, Image.open(new/f'turn_{turn}.png') as b:
                if image_hash(a) != image_hash(b):
                    raise ValueError('Formal bare path differs from diagnostic bare output')
            matched.append(turn)
        if source_hashes() != pins:
            raise ValueError('Source changed during acceptance')
        write_json(output/'summary.json', {'status':'passed', 'protocol':PROTOCOL_VERSION,
            'comparison': {'identical_pixels':True, 'benchmarks':comparisons}, 'turns':counts,
            'diagnostic_bare_exact_pixel_match':{'session':'cm/05447e326032ca20', 'turns':matched},
            'scope':'Original system prompt and full interleaved raw history; no added framing or labels. DP2 none versus prefix plus attention; engineering parity only.'})
        write_json(output/'status.json', {'state':'completed', 'protocol':PROTOCOL_VERSION})
    except BaseException as exc:
        write_json(output/'status.json', {'state':'failed', 'error':f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    main()

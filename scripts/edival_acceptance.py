#!/usr/bin/env python3
"""Single-GPU EdiVal smoke: true history, bare prompt and cache/attention parity."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

from lance_mice.acceptance import compare_runs, validate_run
from lance_mice.dataset import DEFAULT_ROOTS, load_samples
from lance_mice.edival_dataset import source_identity
from lance_mice.images import image_hash
from lance_mice.protocol import PROTOCOL_VERSION
from lance_mice.runner import write_json

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    paths = [p for p in (ROOT / 'src/lance_mice').glob('*.py') if not p.name.startswith('._')]
    paths += [ROOT / 'pyproject.toml', Path(__file__).resolve()]
    paths += [ROOT / 'scripts' / name for name in ('full_edival_job.py', 'validate_edival_full.py',
                                                   'run_edival_job.sh')]
    paths += [ROOT / 'configs/edival_dataset.json', ROOT / 'scripts/edival_data_check.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_ROOTS['edival'])
    parser.add_argument('--gpu', choices=('0', '1'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    memory = int(subprocess.check_output(['nvidia-smi', '-i', args.gpu,
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if memory > 64:
        raise ValueError(f'GPU {args.gpu} is occupied ({memory} MiB); release its burn first')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pins = source_hashes()
    samples = load_samples(args.dataset, 'smoke', 'edival')
    data_identity = source_identity(args.dataset)
    write_json(output / 'source_hashes.json', pins)
    runs, peaks, attention = {}, [], []
    try:
        for mode in ('none', 'prefix_attention'):
            run = output / mode
            write_json(output / 'status.json', {'state': 'running', 'mode': mode, 'gpu': args.gpu})
            command = [sys.executable, '-B', '-m', 'lance_mice.edival',
                '--dataset', str(args.dataset), '--profile', 'single', '--gpus', args.gpu,
                '--cache-mode', 'none' if mode == 'none' else 'prefix', '--audit',
                '--output', str(run)]
            if mode == 'prefix_attention':
                command.append('--save-attention')
            with (output / f'{mode}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = validate_run(run)
            manifest = json.loads((run / 'run.json').read_text())
            if manifest['settings']['settings']['resolution'] != 512:
                raise ValueError('EdiVal acceptance requires resolution 512')
            for sample in samples:
                with Image.open(run / sample.session_id / 'turn_0_input.png') as saved:
                    if image_hash(saved) != image_hash(sample.read_source()):
                        raise ValueError(f'Saved source differs: {sample.session_id}')
            for key, row in rows.items():
                backend = row['backend']
                with Image.open(run / f'{key}.png') as result:
                    if result.size != (512, 512) or backend['size'] != [512, 512]:
                        raise ValueError(f'Generated resolution differs: {key}')
                if any(s.get('role') == 'framing' or
                       (s.get('role') == 'label' and (s['text'] or s['tokens'][0] != s['tokens'][1]))
                       for s in backend['positive_trace']):
                    raise ValueError(f'Extra history prompt text reached the model: {key}')
                if mode == 'prefix_attention':
                    if row['turn'] > 1 and not backend['prefix_segments_reused']:
                        raise ValueError(f'Prefix cache was not reused: {key}')
                    attention.append(backend['attention'])
                peaks.append(backend['peak_memory_bytes'])
            worker = json.loads((run / 'worker_0.json').read_text())
            expected = [[s.session_id, t] for s in samples for t in range(1, 4)]
            if not worker['completed'] or worker['gpu'] != args.gpu or worker['generated_turns'] != expected:
                raise ValueError('Single-GPU worker coverage differs')
            runs[mode] = {'turns': len(rows), 'worker': worker}
        comparison = compare_runs(output / 'none', output / 'prefix_attention')
        if not comparison['identical_pixels']:
            raise ValueError('Prefix cache or attention observation changed pixels')
        if source_hashes() != pins or source_identity(args.dataset) != data_identity:
            raise ValueError('Source or dataset changed during acceptance')
        summary = {
            'status': 'passed', 'benchmark': 'edival', 'protocol': PROTOCOL_VERSION,
            'resolution': 512,
            'profile': 'single', 'gpu': args.gpu, 'sessions': len(samples),
            'turns_per_run': comparison['turns'], 'generated_turns': 2 * comparison['turns'],
            'attention_files': len(attention), 'comparison': comparison,
            'max_attention_reference_error': max(a['max_reference_absolute_error'] for a in attention),
            'max_attention_probability_sum_error': max(a['max_probability_sum_error'] for a in attention),
            'peak_torch_allocated_bytes': max(peaks), 'runs': runs,
            'source_hashes': pins, 'dataset_source': data_identity,
            'scope': 'Single-GPU engineering acceptance only; no DP2 runtime, full inference or capability scoring.',
        }
        write_json(output / 'summary.json', summary)
        write_json(output / 'status.json', {'state': 'completed', 'gpu': args.gpu})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as exc:
        write_json(output / 'status.json', {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    main()

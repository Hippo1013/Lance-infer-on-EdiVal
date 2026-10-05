#!/usr/bin/env python3
"""CPU provenance audit of a complete 512-pixel EdiVal inference plus attention."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from lance_mice.acceptance import validate_run
from lance_mice.dataset import DEFAULT_ROOTS, load_samples
from lance_mice.edival_dataset import source_identity
from lance_mice.images import image_hash
from lance_mice.protocol import digest, PROTOCOL_VERSION
from lance_mice.runner import write_json
from edival_acceptance import source_hashes


def audit(output: Path, dataset: Path) -> dict:
    run = output / 'run'
    manifest = json.loads((run / 'run.json').read_text())
    frozen = json.loads((output / 'source_hashes.json').read_text())
    if source_hashes() != frozen:
        raise ValueError('Runtime source differs from launch pins')
    if manifest['benchmark'] != 'edival' or manifest['profile'] != 'single':
        raise ValueError('Expected single-GPU EdiVal manifest')
    if manifest['settings']['protocol'] != PROTOCOL_VERSION or manifest['settings']['settings']['resolution'] != 512:
        raise ValueError('Expected bare-v2 at 512 pixels')
    if manifest['settings']['settings']['cache_mode'] != 'prefix' or not manifest.get('attention'):
        raise ValueError('Expected prefix caching and attention')
    if manifest['dataset_source'] != source_identity(dataset):
        raise ValueError('Dataset release identity changed')
    samples = load_samples(dataset, 'all', 'edival')
    expected = [[s.session_id, list(s.instructions)] for s in samples]
    if len(samples) != 572 or manifest['samples'] != expected:
        raise ValueError('Full EdiVal release coverage differs')
    rows = validate_run(run)
    allowed = {'session.json', 'turn_0_input.png'} | {
        f'turn_{t}.{ext}' for t in range(1, 4) for ext in ('png', 'json', 'attention.npz')}
    errors, sum_errors, peak = [], [], 0
    if {p.name for p in (run / 'edival').iterdir()} != {s.sample_id for s in samples}:
        raise ValueError('Unexpected EdiVal session directories')
    for sample in samples:
        folder = run / sample.session_id
        spec = json.loads((folder / 'session.json').read_text())
        source = sample.read_source()
        if source.size != (512, 512):
            raise ValueError('Source resolution differs')
        expected_fingerprint = digest([sample.session_id, sample.instructions, image_hash(source),
                                       manifest['settings'], manifest['fingerprint']])
        if (spec['fingerprint'] != expected_fingerprint or spec['instructions'] != list(sample.instructions)
                or spec['source_hash'] != image_hash(source) or spec['source'] != str(sample.image)
                or spec['source_member'] != sample.image_member):
            raise ValueError(f'Source or session identity mismatch: {sample.session_id}')
        with Image.open(folder / 'turn_0_input.png') as saved:
            if saved.size != (512, 512) or image_hash(saved) != image_hash(source):
                raise ValueError(f'Archived source differs: {sample.session_id}')
        if {p.name for p in folder.iterdir()} != allowed:
            raise ValueError(f'Missing, extra or orphan session files: {sample.session_id}')
        for turn in range(1, 4):
            row = rows[f'{sample.session_id}/turn_{turn}']
            backend = row['backend']
            if row['session_fingerprint'] != expected_fingerprint or backend['session_id'] != sample.session_id:
                raise ValueError('Turn belongs to a different session')
            with Image.open(folder / f'turn_{turn}.png') as image:
                if image.size != (512, 512) or backend['size'] != [512, 512]:
                    raise ValueError('Output resolution differs')
            if turn > 1 and not backend['prefix_segments_reused']:
                raise ValueError('Prefix cache was not used')
            if any(s.get('role') == 'framing' or
                   (s.get('role') == 'label' and (s['text'] or s['tokens'][0] != s['tokens'][1]))
                   for s in backend['positive_trace']):
                raise ValueError('Extra history prompt text reached the model')
            a = backend['attention']
            errors.append(a['max_reference_absolute_error'])
            sum_errors.append(a['max_probability_sum_error'])
            peak = max(peak, backend['peak_memory_bytes'])
    worker = json.loads((run / 'worker_0.json').read_text())
    expected_turns = [[s.session_id, t] for s in samples for t in range(1, 4)]
    if not worker['completed'] or worker['turns'] != 1716 or worker['generated_turns'] != expected_turns:
        raise ValueError('Worker coverage differs')
    if source_hashes() != frozen or manifest['dataset_source'] != source_identity(dataset):
        raise ValueError('Runtime source or dataset changed during audit')
    return {'status': 'passed', 'benchmark': 'edival', 'protocol': PROTOCOL_VERSION,
            'resolution': 512, 'sessions': len(samples), 'turns': len(rows), 'attention_files': len(errors),
            'gpu': worker['gpu'], 'max_attention_reference_error': max(errors),
            'max_attention_probability_sum_error': max(sum_errors), 'peak_torch_allocated_bytes': peak,
            'source_files': len(frozen), 'run_fingerprint': manifest['fingerprint'],
            'checks': ['all 572 canonical sources and original instruction chains',
                       '512-pixel saved sources and all generated outputs',
                       'true RGB history, bare prompt and current-only CFG removal',
                       'session/turn seeds, KV cache reuse and complete single-GPU coverage',
                       '1716 attention hashes, shapes, groups and probability checks',
                       'frozen runtime source and original CSV/ZIP identities'],
            'capability_scoring': 'not run'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_ROOTS['edival'])
    args = parser.parse_args()
    report = audit(args.output, args.dataset)
    write_json(args.output / 'validation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

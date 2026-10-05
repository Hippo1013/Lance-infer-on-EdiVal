#!/usr/bin/env python3
"""CPU-only audit of one full MICE pass; never modify inference artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image, PngImagePlugin

from lance_mice.acceptance import validate_run
from lance_mice.dataset import DEFAULT_ROOTS, load_samples, shard_samples
from lance_mice.images import image_hash
from lance_mice.protocol import digest, protocol_manifest
from lance_mice.runner import write_json

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(output: Path, dataset: Path) -> dict:
    run = output / 'run'
    manifest = json.loads((run / 'run.json').read_text())
    samples = load_samples(dataset, 'all')
    require(len(samples) == 720 and sum(len(s.instructions) for s in samples) == 2160,
            'Unexpected MICE release size')
    require(manifest['samples'] == [[s.session_id, list(s.instructions)] for s in samples],
            'Run differs from original dataset sessions or raw instructions')
    require(manifest['profile'] == 'dp2' and manifest['audit']
            and manifest['attention'] == 'target-group-mass-v1', 'Unexpected run mode')
    expected = {f'{s.session_id}/turn_{t}' for s in samples for t in range(1, 4)}
    for suffix in ('.png', '.json', '.attention.npz'):
        actual = {str(p.relative_to(run))[:-len(suffix)]
                  for p in run.glob(f'*/*/turn_*{suffix}') if not p.name.startswith('turn_0_')}
        require(actual == expected, f'Missing or extra artifacts: {suffix}')
    require(not any(run.rglob('*.part')), 'Uncommitted artifacts remain')
    generated = []
    for worker in (0, 1):
        record = json.loads((run / f'worker_{worker}.json').read_text())
        shard = shard_samples(samples, worker, 2)
        turns = [[s.session_id, t] for s in shard for t in range(1, 4)]
        require(record['completed'] and record['gpu'] == str(worker)
                and record['worker'] == worker and record['turns'] == len(turns)
                and record['sessions'] == [s.session_id for s in shard]
                and record['generated_turns'] == turns, f'Worker {worker} mismatch')
        generated.extend(f'{s}/turn_{t}' for s, t in record['generated_turns'])
    require(len(generated) == len(set(generated)) == len(expected)
            and set(generated) == expected, 'Duplicate or missing worker turns')
    pins = json.loads((output / 'source_hashes.json').read_text())
    for name, pin in pins.items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == pin,
                f'Inference or acceptance source drift: {name}')

    # One benchmark JPEG has a 2.62 MiB ICC profile retained in the saved PNG.
    # Bound metadata decoding to 4 MiB in this CPU audit, restoring the default
    # afterward. Pixel decoding is unchanged; no image or NPZ is rewritten.
    old_limit = PngImagePlugin.MAX_TEXT_CHUNK
    try:
        PngImagePlugin.MAX_TEXT_CHUNK = 4 * 1024 * 1024
        rows = validate_run(run)
        require(set(rows) == expected, 'Validation coverage mismatch')
        for sample in samples:
            folder = run / sample.session_id
            spec = json.loads((folder / 'session.json').read_text())
            with Image.open(sample.image) as image:
                source_hash = image_hash(image)
            with Image.open(folder / 'turn_0_input.png') as image:
                require(image_hash(image) == source_hash, 'Saved source pixels differ')
            fingerprint = digest([sample.session_id, sample.instructions, source_hash,
                                  manifest['settings'], manifest['fingerprint']])
            require(spec == {'fingerprint': fingerprint, 'session_id': sample.session_id,
                            'source': str(sample.image), 'source_hash': source_hash,
                            'instructions': list(sample.instructions),
                            'settings': manifest['settings']}, 'Session identity mismatch')
            for turn in range(1, 4):
                row = rows[f'{sample.session_id}/turn_{turn}']
                require(row['session_fingerprint'] == fingerprint and row['turn'] == turn
                        and row['protocol'] == protocol_manifest(list(sample.instructions[:turn]), version=manifest['settings']['protocol']),
                        'Saved prompt differs from approved protocol or session')
    finally:
        PngImagePlugin.MAX_TEXT_CHUNK = old_limit
    return {
        'status': 'passed', 'verified_at': datetime.now(timezone.utc).isoformat(),
        'sessions': len(samples), 'turns': len(rows), 'attention_files': len(rows),
        'source_hashes_checked': len(pins), 'png_metadata_limit_bytes': 4 * 1024 * 1024,
        'max_reference_absolute_error': max(
            row['backend']['attention']['max_reference_absolute_error'] for row in rows.values()),
        'max_probability_sum_error': max(
            row['backend']['attention']['max_probability_sum_error'] for row in rows.values()),
        'validator': str(Path(__file__).relative_to(ROOT)),
        'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks': ['dataset sources and raw instructions', 'exact file and DP2 coverage',
                   'approved prompt and session identities', 'actual GPU traces and image history',
                   'CFG and seeds', 'attention hashes, groups, dimensions and probability checks',
                   'original inference source hashes'],
        'capability_scoring': 'not run',
        'note': 'CPU-only post-validation. Original images, attention and inference sources unchanged.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_ROOTS['mice'])
    args = parser.parse_args()
    report = audit(args.output.resolve(), args.dataset)
    write_json(args.output / 'validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

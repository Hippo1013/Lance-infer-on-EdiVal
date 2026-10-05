#!/usr/bin/env python3
"""Check ImgEdit metadata and one fixed 2/3/3-turn DP2 + prefix smoke batch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image
from lance_mice.acceptance import validate_run
from lance_mice.dataset import DEFAULT_ROOTS, load_samples, shard_samples
from lance_mice.protocol import history_segments, protocol_manifest
from lance_mice.runner import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=DEFAULT_ROOTS['imgedit'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', default='0,1')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty acceptance output directory')
    all_samples = load_samples(args.dataset, 'all', 'imgedit')
    samples = load_samples(args.dataset, 'smoke', 'imgedit')
    sources = []
    for sample in all_samples:
        with Image.open(sample.image) as im:
            im.load()
            sources.append({'session': sample.session_id, 'size': im.size,
                            'turns': len(sample.instructions)})
    write_json(args.output / 'data.json', {'sessions': len(sources),
        'turns': sum(x['turns'] for x in sources), 'decoded_sources': sources})
    run = args.output / 'smoke_dp2_prefix'
    command = [sys.executable, '-m', 'lance_mice.imgedit', '--dataset', str(args.dataset),
               '--gpus', args.gpus, '--audit', '--output', str(run)]
    subprocess.run(command, check=True)
    rows = validate_run(run)
    for sample in samples:
        for turn in range(1, len(sample.instructions) + 1):
            row = rows[f'{sample.session_id}/turn_{turn}']
            instructions = list(sample.instructions[:turn])
            assert row['protocol'] == protocol_manifest(instructions)
            assert row['instruction'] == instructions[-1]
            assert row['backend']['session_id'] == sample.session_id
            expected = []
            for s in history_segments(instructions):
                if s.kind == 'image':
                    expected.extend([(k, s.image_index) for k in ('vit', 'vae')])
                else:
                    expected.append(('text', s.role, s.turn, s.text))
            observed = [(x['kind'], x['image_index']) if x['kind'] in ('vit', 'vae')
                        else ('text', x['role'], x['turn'], x['text'])
                        for x in row['backend']['positive_trace']]
            assert observed == expected, (sample.session_id, turn, 'interleaving differs')
        assert not (run / sample.session_id / f'turn_{len(sample.instructions)+1}.png').exists()
    workers = []
    for i in range(2):
        worker = json.loads((run / f'worker_{i}.json').read_text())
        assigned = shard_samples(samples, i, 2)
        expected = [[s.session_id, t] for s in assigned for t in range(1, len(s.instructions)+1)]
        assert worker['completed'] and worker['generated_turns'] == expected
        assert worker['gpu'] == args.gpus.split(',')[i]
        workers.append(worker)
    assert workers[0]['pid'] != workers[1]['pid']
    report = {'status': 'passed', 'benchmark': 'imgedit', 'sessions': len(samples), 'turns': len(rows),
        'profile': 'dp2', 'cache': 'prefix', 'workers': workers,
        'checks': ['all source images decode', 'category namespaces and two/three-turn lengths',
                   'each GPU receives the exact ordered full-history segments',
                   'raw instructions and no evaluation answers', 'actual saved outputs feed later turns',
                   'current-only CFG removal and session/turn seeds', 'complete session sharding'],
        'scope': 'Engineering smoke only; no throughput/cache comparison, full inference or official capability scoring.'}
    write_json(args.output / 'summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

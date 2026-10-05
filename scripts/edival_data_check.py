#!/usr/bin/env python3
"""CPU-only EdiVal release audit: prefixes, CSV coverage, ZIP CRC and RGB decode."""

import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path
from zipfile import ZipFile

from lance_mice.dataset import DEFAULT_ROOTS, load_samples, shard_samples
from lance_mice.edival_dataset import CSV_NAME, ZIP_NAME, source_identity


def audit_dataset(root: Path) -> dict:
    samples = load_samples(root, 'all', 'edival')
    sizes, modes = Counter(), Counter()
    for sample in samples:
        image = sample.read_source()
        sizes[f'{image.width}x{image.height}'] += 1
        modes[image.mode] += 1
        image.close()
    with ZipFile(root / ZIP_NAME) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f'EdiVal ZIP CRC failed: {bad_member}')
        names = {n for n in archive.namelist() if not n.endswith('/')}
    task_types, csv_rows = Counter(), 0
    with (root / CSV_NAME).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            # Evaluation labels are used only for this data audit, never for prompts.
            task_types[ast.literal_eval(row['task_type'])[-1]] += 1
            csv_rows += 1
    return {
        'status': 'passed', 'benchmark': 'edival', 'csv_rows': csv_rows,
        'sessions': len(samples), 'turns': sum(len(s.instructions) for s in samples),
        'archive_images': len(names),
        'unreferenced_images': len(names - {s.image_member for s in samples}),
        'source_sizes': dict(sizes), 'source_modes': dict(modes),
        'task_types': dict(sorted(task_types.items())),
        'dp2_sessions': [len(shard_samples(samples, i, 2)) for i in range(2)],
        'dataset_source': source_identity(root),
        'checks': ['unique complete three-turn sessions', 'exact cumulative prefixes',
                   'all CSV-referenced images present and decoded', 'ZIP CRC',
                   'coverage from CSV rather than archive members'],
        'scope': 'Data audit only; no model inference or capability scoring.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_ROOTS['edival'])
    parser.add_argument('--release-manifest', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'configs/edival_dataset.json')
    args = parser.parse_args()
    report = audit_dataset(args.dataset)
    expected = json.loads(args.release_manifest.read_text(encoding='utf-8'))
    for key in ('sessions', 'turns', 'csv_rows', 'archive_images', 'unreferenced_images'):
        if report[key] != expected[key]:
            raise ValueError(f'EdiVal release count differs: {key}')
    for name, sha256 in expected['sha256'].items():
        if report['dataset_source'][name]['sha256'] != sha256:
            raise ValueError(f'EdiVal release file differs: {name}')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

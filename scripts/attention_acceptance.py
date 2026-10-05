#!/usr/bin/env python3
"""Verify full-context attention export without changing generated pixels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from lance_mice.acceptance import compare_runs, validate_run
from lance_mice.runner import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=Path('outputs/mice/acceptance_20261003_steps67/correctness_single'))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.output / 'smoke_dp2_prefix'
    subprocess.run([sys.executable, '-m', 'lance_mice.runner', '--profile', 'dp2', '--gpus', '0,1',
        '--cache-mode', 'prefix', '--audit', '--save-attention', '--output', str(run)], check=True)
    rows = validate_run(run)
    comparison = compare_runs(args.reference, run)
    if not comparison['identical_pixels']:
        raise ValueError('Attention observation changed generated pixels')
    observations = {key: row['backend']['attention'] for key, row in rows.items()}
    write_json(args.output / 'summary.json', {'status': 'passed', 'turns': len(rows),
        'comparison': comparison, 'observations': observations,
        'max_reference_absolute_error': max(x['max_reference_absolute_error'] for x in observations.values()),
        'max_probability_sum_error': max(x['max_probability_sum_error'] for x in observations.values()),
        'scope': 'Positive generation branch, every diffusion step/layer/head; target-query mean mass over the complete key set. Original images unchanged. No capability scoring.'})
    print(json.dumps({'status':'passed','turns':len(rows),'identical_pixels':True}))


if __name__ == '__main__':
    main()

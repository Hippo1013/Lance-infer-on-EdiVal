#!/usr/bin/env python3
"""Disjoint MICE session shards with one shared, immutable vote identity."""
import argparse
import fcntl
from pathlib import Path
from lance_mice import scoring_qwen_once as profile, scoring_pending as pending
from lance_mice.scoring import core, runner
from lance_mice.images import image_hash
from sixrun_queue import directory_lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['metrics', 'judge'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--inference', type=Path, required=True)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, default=4)
    a = p.parse_args()
    if not 0 <= a.shard < a.shards:
        raise ValueError('Invalid shard')
    profile.configure()
    a.dataset = Path('/home/chs/dataset/MICE-Bench')
    a.model_root = Path('/home/chs/model')
    a.allow_full, a.retry_errors, a.judge = True, True, 'qwen'
    manifest = runner.load_manifest(a)
    # Filter only after full identity validation; never change the vote fingerprint.
    manifest = {**manifest, 'records': manifest['records'][a.shard::a.shards]}
    original_write = core.write
    def scoped_write(path, value):
        path = Path(path)
        if path.name == 'judge_progress.json':
            path = path.with_name(f'judge_progress_{a.shard}.json')
        return original_write(path, value)
    core.write = scoped_write
    # Detection keys can also coincide for common source images.
    from lance_mice.scoring.metrics import Detector, detection_key
    original_detect = Detector.__call__
    def locked_detect(self, image, target, threshold=.3, return_all=False, delete_large_box=False):
        key = detection_key(image, target, threshold, return_all, delete_large_box)
        folder = a.output / 'detection_locks'
        folder.mkdir(exist_ok=True)
        with directory_lock(folder / key):
            return original_detect(self, image, target, threshold, return_all, delete_large_box)
    Detector.__call__ = locked_detect
    # Shared original source images can generate the same request in two shards.
    # Lock the entire cache lookup/generation/first-response commit transaction.
    original_ask = pending.PreservedJudge.ask
    def locked_ask(self, images, prompt, kind):
        key = core.request_identity([image_hash(im.convert('RGB')) for im in images], prompt, self.vote)
        folder = a.output / 'request_locks'
        folder.mkdir(exist_ok=True)
        with directory_lock(folder / key):
            return original_ask(self, images, prompt, kind)
    pending.PreservedJudge.ask = locked_ask
    with (a.output / f'.{a.stage}_{a.shard}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.stage == 'metrics':
            runner.metrics_stage(a, manifest)
        else:
            pending.judge_stage(a, manifest)
    core.write(a.output / f'{a.stage}_shard_{a.shard}.json', {
        'status': 'completed', 'sessions': len(manifest['records']),
        'turns': 3 * len(manifest['records']), 'fingerprint': manifest['fingerprint'],
        'record_ids': [r['session_id'] for r in manifest['records']]})


if __name__ == '__main__':
    main()

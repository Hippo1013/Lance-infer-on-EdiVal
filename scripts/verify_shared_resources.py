#!/usr/bin/env python3
"""Verify fresh shared copies against existing local benchmark/model sources."""
from pathlib import Path
from sixrun_queue import sha, write, now


def compare(source, destination):
    originals = {str(p.relative_to(source)): p for p in source.rglob('*') if p.is_file()}
    copies = {str(p.relative_to(destination)): p for p in destination.rglob('*') if p.is_file()}
    if originals.keys() != copies.keys():
        raise ValueError('Shared file inventory differs: '+str(source))
    pins = {}
    for name, path in originals.items():
        target = copies[name]
        if path.stat().st_size != target.stat().st_size:
            raise ValueError('Shared file size mismatch: '+str(target))
        expected = sha(path)
        if sha(target) != expected:
            raise ValueError('Shared content mismatch: '+str(target))
        pins[name] = {'sha256': expected, 'bytes': path.stat().st_size}
    return pins


if __name__ == '__main__':
    shared = Path('/media/damoxing/tangzecong/chs_umm_shared')
    pins = {}
    for family, name in [('model', 'Lance'), ('dataset', 'EdiVal'), ('dataset', 'MICE-Bench'),
                         ('dataset', 'ImgEdit-Bench-Multi-Turn')]:
        pins[f'{family}/{name}'] = compare(Path('/home/chs') / family / name, shared / family / name)
        print('VERIFIED', family, name, flush=True)
    write(shared / 'setup/inference_verified.json', {'status': 'passed', 'at': now(), 'resources': pins})
    for name in ['Qwen3.6-27B', 'DINOv3-ViT-B-16', 'DINOv3-ViT-L-16', 'GroundingDINO-SwinT-OGC',
                 'HPSv3', 'RAHF', 'Qwen2-VL-7B-Instruct', 'bert-base-uncased', 't5-base', 'vit-large-patch16-384']:
        pins[f'model/{name}'] = compare(Path('/home/chs/model') / name, shared / 'model' / name)
        print('VERIFIED model', name, flush=True)
    write(shared / 'setup/all_verified.json', {'status': 'passed', 'at': now(), 'resources': pins})

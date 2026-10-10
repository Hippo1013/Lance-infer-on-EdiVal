#!/usr/bin/env python3
"""Independently bind completed collection to original pixels and frozen code.

No inference or attention metrics are imported. Evidence stays on its producer.
"""
import argparse
import hashlib
import importlib.metadata
import json
import socket
import sys
import time
from pathlib import Path
from PIL import Image

ROOT = Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal')
TASK = 'edival_semantic_20261009_v2'
HOSTS = {'a800_0': 'aibox-r61097f954fe-7d74d99d65-2zzkn',
         'a800_1': 'aibox-r7df77faf487-f8c468557-b9c9s'}
EXPECTED = {'a800_0': [140, 139], 'a800_1': [147, 146]}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pixels(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        digest = hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode())
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=HOSTS, required=True)
    args = parser.parse_args()
    assert socket.gethostname() == HOSTS[args.host]
    collection = ROOT / 'outputs/attention_recollection' / TASK
    original = ROOT / 'outputs/sixrun_20261007/edival_chat/run/edival'
    runtime = ROOT / 'runtime' / (TASK + '_r2')
    manifest = json.loads((runtime / 'recollection_manifest.json').read_text())
    assert manifest['observation_format'] == 'target-token-region-stats-v2'
    for relative, digest in manifest['files'].items():
        assert sha(runtime / relative) == digest, relative
    gate = json.loads((collection / 'gates/pilot_local.json').read_text())
    peer = json.loads((collection / 'gates/pilot_peer.json').read_text())
    assert gate['status'] == peer['status'] == 'passed'
    original_ids = sorted([p.name for p in original.iterdir()
                           if (p / 'session.json').exists()], key=int)
    source_records, completion_hashes, dispatch_hashes = [], {}, {}
    maximum_reference_error = 0.0
    for gpu in [0, 1]:
        stage = collection / f'full_gpu{gpu}'
        completion = json.loads((stage / 'completion.json').read_text())
        dispatch_path = collection / 'dispatch' / f'full_gpu{gpu}.json'
        dispatch = json.loads(dispatch_path.read_text())
        assert dispatch['status'] == 'passed' and dispatch['exit_code'] == 0
        assert dispatch['temporary_directory_removed'] and dispatch['burn_restart_sent']
        assert not (stage / 'failure.json').exists()
        assert completion['status'] == 'passed' and not completion['pilot']
        assert completion['sessions'] == EXPECTED[args.host][gpu]
        assert completion['turns'] == 3 * completion['sessions']
        assert completion['hostname'] == HOSTS[args.host]
        assert completion['runtime_manifest_sha256'] == sha(runtime / 'recollection_manifest.json')
        assert completion['all_outputs_exact'] and completion['all_real_identity_checks_passed']
        for name, digest in completion['runtime_sources'].items():
            assert sha(runtime / 'src/lance_mice' / name) == digest
        selection = json.loads((stage / 'selection.json').read_text())
        assert selection['sessions'] == original_ids[gpu::2]
        assert len(completion['records']) == completion['turns']
        selected_ids = set(selection['sessions'])
        seen = set()
        for record in completion['records']:
            sid, turn = record['session_id'], record['turn']
            index = sid.split('/')[-1]
            assert index in selected_ids and turn in [1, 2, 3]
            assert (sid, turn) not in seen
            seen.add((sid, turn))
            path = stage / 'run' / index / 'observed' / f'turn_{turn}.json'
            receipt = json.loads(path.read_text())
            source_path = original / index / f'turn_{turn}.json'
            source = json.loads(source_path.read_text())
            session = json.loads((original / index / 'session.json').read_text())
            assert sha(source_path) == receipt['original_turn_json_sha256']
            assert receipt['session_fingerprint'] == session['fingerprint']
            assert receipt['source_model_settings'] == session['settings']
            assert receipt['instruction'] == session['instructions'][turn - 1]
            assert receipt['mode'] == record['mode'] == 'observed'
            assert receipt['runtime_sources'] == completion['runtime_sources']
            assert receipt['runtime_manifest_sha256'] == completion['runtime_manifest_sha256']
            history = [original / index / 'turn_0_input.png']
            history += [original / index / f'turn_{i}.png' for i in range(1, turn)]
            assert [pixels(p) for p in history] == source['input_hashes'] == receipt['input_hashes']
            actual = pixels(path.with_suffix('.png'))
            assert actual == pixels(original / index / f'turn_{turn}.png')
            assert actual == source['output_hash'] == receipt['output_hash'] == record['output_hash']
            backend = receipt['backend']
            assert backend['image_hashes'] == source['input_hashes']
            assert backend['seed'] == source['backend']['seed']
            audit = backend['cache_identity_audit']
            assert audit['status'] == 'passed' and audit['layers'] == [0, 35]
            assert all(audit['checks'][f'target_query:layer{layer}'] == 30 for layer in [0, 35])
            attention = backend['attention']
            assert attention['version'] == 'target-token-region-stats-v2'
            assert attention['branch'] == 'positive'
            assert attention['sha256'] == record['attention_sha256']
            maximum_reference_error = max(maximum_reference_error, attention['max_reference_absolute_error'])
            source_records.append({'session_id': sid, 'turn': turn, 'receipt_sha256': sha(path),
                                   'output_rgb_sha256': actual, 'attention_sha256': attention['sha256']})
        assert len(seen) == completion['turns']
        completion_hashes[f'full_gpu{gpu}'] = sha(stage / 'completion.json')
        dispatch_hashes[f'full_gpu{gpu}'] = sha(dispatch_path)
    packages = {name: importlib.metadata.version(name)
                for name in ['torch', 'numpy', 'pillow', 'transformers', 'vllm', 'vllm-omni']}
    environment = {'python': sys.version, 'executable': sys.executable, 'packages': packages}
    assert len(source_records) == 3 * len(original_ids)
    evidence = {'task_id': TASK, 'status': 'passed', 'hostname': socket.gethostname(),
                'sessions': len(original_ids), 'turns': len(source_records),
                'completion_sha256': completion_hashes, 'dispatch_sha256': dispatch_hashes,
                'runtime_manifest_sha256': sha(runtime / 'recollection_manifest.json'),
                'runtime_sources': completion['runtime_sources'], 'environment': environment,
                'max_reference_absolute_error': maximum_reference_error,
                'checks': ['original fixed history pixels', 'saved generated PNGs versus original RGB pixels',
                           'complete unique shard selection', 'every real target query audit',
                           'all frozen runtime files and original settings identities'],
                'records': source_records, 'validator_sha256': sha(Path(__file__)), 'at_unix': time.time()}
    destination = collection / 'validation/recollection_independent_completion.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({key: evidence[key] for key in ['status', 'sessions', 'turns', 'environment',
                                                   'max_reference_absolute_error']}))


if __name__ == '__main__':
    main()

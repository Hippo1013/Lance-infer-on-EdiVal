#!/usr/bin/env python3
"""Verify the scientific freeze and rendered publication without moving data."""
import gzip
import hashlib
import json
import re
import socket
import time
from pathlib import Path

ROOT = Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal')
TASK = 'edival_semantic_20261009_v2'
SOURCE = ROOT / 'outputs/attention_analysis' / TASK
REPORT = ROOT / 'outputs/attention_reports' / TASK
COUNTS = {'group_turn': 15444, 'image_combined_turn': 3432, 'category_turn': 12012,
          'category_modality_turn': 6864, 'object_trajectories': 13728,
          'paired_changes': 9152, 'concentration_turn': 10296, 'trajectory_matrix': 20592,
          'text_token_profiles': 36312, 'image_region_profiles': 439296,
          'within_type_pairs': 9152, 'image_marker_turn': 13728, 'within_type_summary': 64}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_manifest(path):
    manifest = read(path)
    seen = set()
    for row in manifest['files']:
        relative = row['path']
        assert relative not in seen
        seen.add(relative)
        p = SOURCE / relative
        assert p.is_relative_to(SOURCE)
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], relative
    return len(seen)


def main():
    assert socket.gethostname() == 'aibox-r61097f954fe-7d74d99d65-2zzkn'
    completion = read(SOURCE / 'completion.json')
    assert completion['status'] == 'passed' and completion['issues'] == []
    assert completion['input_sessions'] == 572 and completion['input_turns'] == 1716
    artifact = SOURCE / completion['artifact_manifest_path']
    assert sha(artifact) == completion['artifact_manifest_sha256']
    candidate = SOURCE / 'metadata/candidate_manifest.json'
    assert sha(candidate) == completion['independently_validated_candidate_sha256']
    candidate_files = check_manifest(candidate)
    artifact_files = check_manifest(artifact)
    for name in ['validation', 'self_validation', 'independent_samples', 'numerical_tests',
                 'input_audit', 'within_type_checks', 'completion_checks']:
        assert read(SOURCE / 'validation' / f'{name}.json')['status'] == 'passed', name
    validation = read(SOURCE / 'validation/self_validation.json')
    assert validation['compressed_tables_read'] == 17
    assert validation['compressed_arrays_read'] == 1716
    for name, count in COUNTS.items():
        assert validation['table_rows'][name] == count, name
    assert len(list((SOURCE / 'arrays').rglob('*.npz'))) == 1716
    with gzip.open(SOURCE / 'metadata/source_manifest.jsonl.gz', 'rt') as stream:
        sources = [json.loads(line) for line in stream]
    assert len(sources) == len({s['source_id'] for s in sources}) == 1716
    assert len({s['session_id'] for s in sources}) == 572
    assert len({s['input_hashes'][0] for s in sources}) == 570
    assert all(s['sha256_verified'] and s['index_identity_verified'] for s in sources)
    assert {s['attention_version'] for s in sources} == {'target-token-region-stats-v2'}
    assert {s['source_run'] for s in sources} == {TASK}
    code = read(SOURCE / 'metadata/code_manifest.json')
    for name, digest in code['files'].items():
        assert sha(SOURCE / 'metadata/analysis_source' / name) == digest
        assert sha(ROOT / 'analysis/edival_attention_semantic_v2' / name) == digest
    ready = read(REPORT / 'publication_ready.json')
    assert ready['status'] == 'ready_for_local_review'
    assert ready['source_artifact_manifest_sha256'] == sha(artifact)
    assert ready['report_sha256'] == sha(REPORT / 'attention分析结果.md')
    assert len(ready['figures']) == 24
    assert sum(name.endswith('.png') for name in ready['figures']) == 12
    assert sum(name.endswith('.svg') for name in ready['figures']) == 12
    for name, digest in ready['figures'].items():
        assert sha(REPORT / name) == digest, name
    markdown = (REPORT / 'attention分析结果.md').read_text()
    links = re.findall(r'!\[[^\]]*\]\(([^)]*)\)', markdown)
    assert len(links) == 12
    assert {Path(link).name for link in links} == {n for n in ready['figures'] if n.endswith('.png')}
    assert all(link.startswith('/Users/hippo/Desktop/research_ws/umm/多轮次图像编辑/exp0_attention/attention分析图表/前三步/') for link in links)
    evidence = {'status': 'passed', 'task_id': TASK, 'artifact_manifest_sha256': sha(artifact),
                'candidate_manifest_sha256': sha(candidate), 'artifact_files': artifact_files,
                'candidate_files': candidate_files, 'source_sessions': 572, 'source_turns': 1716,
                'source_image_clusters': 570, 'tables': 17, 'derived_arrays': 1716,
                'figure_sha256': ready['figures'], 'report_sha256': ready['report_sha256'],
                'validator_sha256': sha(Path(__file__)), 'at_unix': time.time(),
                'pending': ['local visual review', 'full prose and fact review', 'resource closure']}
    (REPORT / 'publication_audit.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({k: v for k, v in evidence.items() if k not in ['figure_sha256']}))


if __name__ == '__main__':
    main()

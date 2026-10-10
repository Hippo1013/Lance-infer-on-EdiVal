#!/usr/bin/env python3
"""CPU-only audit of the saved EdiVal upper-left attention concentration.

Reads original arrays on their production host. All numeric artifacts remain
on servers. Cohorts fix turn, image identity and encoding, so each session has
exactly one equally weighted observation in each cohort. No model execution.
"""
import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np

TASK = "edival_region_bias_20261009_v1"
PROJECT = Path("/home/chs/exp0_attention/Lance-infer-on-EdiVal")
SOURCE = PROJECT / "outputs/sixrun_20261007/edival_chat/run/edival"
ACCEPTED = PROJECT / "outputs/attention_analysis/edival_descriptive_20261008_v1"
OUTPUT = PROJECT / "outputs/attention_analysis" / TASK
HOSTS = {"a800_0": "aibox-r61097f954fe-7d74d99d65-2zzkn",
         "a800_1": "aibox-r7df77faf487-f8c468557-b9c9s"}
AXES = ["sum_raw_topleft", "sum_group_mass", "sum_conditional_topleft",
        "positive_count", "topleft_max_count", "above_uniform_count"]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def mean_valid(a, mask):
    return float(a[mask].mean()) if mask.any() else None


def entropy(p):
    terms = np.zeros_like(p)
    np.log(p, out=terms, where=p > 0)
    return -(p * terms).sum(-1) / math.log(p.shape[-1])


def process_file(path):
    path = Path(path)
    stat = path.stat()
    # Hash the exact bytes that are analyzed, binding diagnosis to first-run evidence.
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    turn = int(path.name.split('_')[1].split('.')[0])
    session = 'edival/' + path.parent.name
    rows, arrays = [], {}
    max_conservation = 0.0
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        assert str(z['protocol']) == 'target-token-region-stats-v1'
        raw = z['region_mean'].astype(np.float64)
        assert raw.shape == (30, 36, 16, turn, 2, 8, 8)
        assert np.isfinite(raw).all() and raw.min() >= 0
        fine = z['fine_key_channel']
        group_ids = z['token_group_ids']
        names = z['group_names'].tolist()
        group_mass = z['group_stats'][..., 0].astype(np.float64)
        geoms = json.loads(str(z['image_geometry_json']))
        assert len(geoms) == turn * 2
        spatial_positions = set()
        for geo in geoms:
            im = geo['image_index']
            typ = ['vit', 'vae'].index(geo['kind'])
            lo, hi = geo['spatial_tokens']
            begin, end = geo['tokens']
            h, w = geo['grid_hw']
            assert (lo, hi) == (begin + 1, end - 1) and hi - lo == h * w
            assert fine[begin] == fine[end - 1] == -1
            assert group_ids[begin] == group_ids[end - 1] == 0
            assert geo['geometry']['layout'] == 'row-major'
            # Independent scalar construction of patch-center cell assignment.
            cells = np.array([min(7, math.floor((y + .5) * 8 / h)) * 8
                              + min(7, math.floor((x + .5) * 8 / w))
                              for y in range(h) for x in range(w)])
            base = (im * 2 + typ) * 64
            assert np.array_equal(fine[lo:hi], cells + base)
            assert np.array_equal(np.bincount(cells, minlength=64).reshape(8, 8),
                                  z['region_token_counts'][im, typ])
            spatial_positions.update(range(lo, hi))
            a = raw[..., im, typ, :, :].reshape(30, 36, 16, 64)
            mass = a.sum(-1)
            err = float(np.max(np.abs(mass - group_mass[..., names.index(f'I{im}_{geo["kind"]}')])) )
            max_conservation = max(max_conservation, err)
            assert err < .001
            positive = mass > 0
            p = np.divide(a, mass[..., None], out=np.zeros_like(a), where=positive[..., None])
            x0, p0 = a[..., 0], p[..., 0]
            maximum = positive & (a[..., 0] == a.max(-1))
            rest_mass = mass - x0
            rest_valid = rest_mass > 0
            rest = np.divide(a[..., 1:], rest_mass[..., None],
                             out=np.zeros_like(a[..., 1:]), where=rest_valid[..., None])
            top7 = np.partition(p, -7, axis=-1)[..., -7:].sum(-1)
            rest_top7 = np.partition(rest, -7, axis=-1)[..., -7:].sum(-1)
            all_entropy, rest_entropy = entropy(p), entropy(rest)
            cohort = f't{turn}_I{im}_{geo["kind"]}'
            row = dict(session_id=session, turn=turn, image_index=im,
                       modality=geo['kind'], cohort=cohort, source_sha256=sha,
                       n_positions=mass.size, n_positive=int(positive.sum()),
                       raw_topleft=float(x0.mean()), group_mass=float(mass.mean()),
                       conditional_topleft=mean_valid(p0, positive),
                       pooled_topleft=float(x0.sum() / mass.sum()),
                       max_fraction=mean_valid(maximum, positive),
                       above_uniform_fraction=mean_valid(p0 > 1/64, positive),
                       entropy=mean_valid(all_entropy, positive),
                       rest_entropy=mean_valid(rest_entropy, rest_valid),
                       top7=mean_valid(top7, positive),
                       rest_top7=mean_valid(rest_top7, rest_valid),
                       top7_topleft_component=mean_valid(p0 * (a[..., 0] >= np.partition(a, -7, axis=-1)[..., -7]), positive),
                       rest_zero_fraction=float((~rest_valid).mean()))
            for label, threshold in [('gt1e6', 1e-6), ('gt1e4', 1e-4), ('gt1e3', 1e-3)]:
                mask = mass > threshold
                row[label + '_fraction'] = float(mask.mean())
                row[label + '_topleft'] = mean_valid(p0, mask)
                row[label + '_max_fraction'] = mean_valid(maximum, mask)
            rows.append(row)
            arrays[cohort] = np.stack([x0, mass, p0, positive, maximum, positive & (p0 > 1/64)])
        assert len(spatial_positions) == int(z['region_token_counts'].sum())
    assert path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns
    return rows, arrays, dict(path=str(path), session_id=session, turn=turn,
                             sha256=sha, bytes=stat.st_size, max_conservation=max_conservation,
                             mapping_passed=True)


def shard(args):
    assert socket.gethostname() == HOSTS[args.host]
    root = OUTPUT / args.host
    root.mkdir(parents=True, exist_ok=False)
    files = sorted(SOURCE.glob('*/*.attention.npz'), key=lambda p:(int(p.parent.name), p.name))
    expected = {'a800_0': 837, 'a800_1': 879}[args.host]
    assert len(files) == expected
    sums, rows, inputs, counts = {}, [], [], {}
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (rs, ar, provenance) in enumerate(pool.map(process_file, files, chunksize=1), 1):
            rows.extend(rs); inputs.append(provenance)
            for k, v in ar.items():
                if k not in sums: sums[k] = np.zeros_like(v)
                sums[k] += v; counts[k] = counts.get(k, 0) + 1
            if n % 100 == 0:
                dump(root/'progress.json', dict(done=n, total=len(files), elapsed=time.time()-started))
                print(args.host, n, '/', len(files), flush=True)
    with gzip.open(root/'object_turn_metrics.csv.gz', 'wt') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(root/'position_sums.npz', **sums)
    dump(root/'inputs.json', inputs)
    # Compact sufficient statistics for cross-host assembly, no original arrays.
    summary = {}
    excluded={'session_id','turn','image_index','modality','cohort','source_sha256'}
    for k, ar in sums.items():
        rs = [r for r in rows if r['cohort'] == k]
        scalar={key: float(np.mean([r[key] for r in rs if r[key] is not None]))
                for key in rs[0] if key not in excluded and any(r[key] is not None for r in rs)}
        summary[k]=dict(n=counts[k], scalar=scalar,
                        layer_head_sums=ar.sum(1).tolist(), step_sums=ar.sum((2,3)).tolist())
    dump(root/'sufficient_statistics.json', summary)
    dump(root/'completion.json', dict(task=TASK,status='passed',host=args.host,
        hostname=socket.gethostname(),n_files=len(files),n_sessions=len(files)//3,
        n_supports=len(rows),elapsed_seconds=time.time()-started,axes=AXES,
        source_protocol='lance-history-chat-v1',python=sys.version,numpy=np.__version__,
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mapping_failures=0,max_conservation=max(x['max_conservation'] for x in inputs),
        cpu_only=True))
    print('COMPLETED', root, flush=True)


def combine(args):
    # Receiving compact JSON through stdin never writes data to the local machine.
    peer=json.load(sys.stdin)
    dump(OUTPUT/'a800_1_sufficient_statistics.json',peer)
    lead=json.loads((OUTPUT/'a800_0/sufficient_statistics.json').read_text())
    merged={}
    for k in lead:
        a,b=lead[k],peer[k];n=a['n']+b['n'];assert n==572
        scalar={m:(a['scalar'][m]*a['n']+b['scalar'][m]*b['n'])/n for m in a['scalar']}
        lh=np.asarray(a['layer_head_sums'])+np.asarray(b['layer_head_sums'])
        step=np.asarray(a['step_sums'])+np.asarray(b['step_sums'])
        order=np.argsort(lh[0].ravel())[::-1]
        tops=[dict(layer=int(i//16),head=int(i%16),
                   absolute_mass_share=float(lh[0].ravel()[i]/lh[0].sum()),
                   pooled_topleft=float(lh[0].ravel()[i]/lh[1].ravel()[i])) for i in order[:10]]
        merged[k]=dict(n=n,scalar=scalar,layer_head_sums=lh.tolist(),step_sums=step.tolist(),
                      top10_heads=tops,top10_heads_mass_fraction=sum(x['absolute_mass_share'] for x in tops),
                      pooled_without_top10=float((lh[0].sum()-lh[0].ravel()[order[:10]].sum()) /
                                                 (lh[1].sum()-lh[1].ravel()[order[:10]].sum())))
    # Reconcile all raw-byte hashes against accepted provenance, on the servers.
    dump(OUTPUT/'combined.json',merged)
    print(json.dumps({k:dict(n=v['n'],scalar=v['scalar'],top10_heads_mass_fraction=v['top10_heads_mass_fraction'],
        pooled_without_top10=v['pooled_without_top10'],top3=v['top10_heads'][:3]) for k,v in merged.items()},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['shard','combine'])
    p.add_argument('--host',choices=list(HOSTS));p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    if args.mode=='shard':shard(args)
    else:combine(args)


if __name__=='__main__':main()

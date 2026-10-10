"""EdiVal v2 offline process/head/query/spatial analysis, with explicit supports.

All input arrays are query-averaged; group standard deviations are across queries.
No generation, scoring or model imports. Nonlinear metrics precede aggregation.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np

LOW = 1e-6
SHAPE = (30, 36, 16)


def mean_valid(a, axis=None):
    a = np.asarray(a, dtype=np.float64)
    valid = np.isfinite(a)
    n = valid.sum(axis=axis)
    sums = np.where(valid, a, 0).sum(axis=axis)
    return np.divide(sums, n, out=np.full(np.shape(sums), np.nan), where=n > 0)


def ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def entropy(p):
    logp = np.zeros_like(p, dtype=np.float64)
    np.log(p, out=logp, where=p > 0)
    return -(np.where(p > 0, p * logp, 0)).sum(-1)


def distribution(x):
    x = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(x < 0) or not x.shape[-1]:
        raise ValueError('Invalid nonnegative support')
    mass = x.sum(-1)
    p = ratio(x, mass[..., None])
    return mass, p


def concentration(x):
    mass, p = distribution(x)
    n = x.shape[-1]
    ordered = np.sort(np.nan_to_num(p), axis=-1)[..., ::-1]
    csum = ordered.cumsum(-1)
    valid = mass > 0
    h = entropy(np.nan_to_num(p))
    n90 = (csum >= .9 - 1e-12).argmax(-1) + 1
    def mask(a):
        return np.where(valid, a, np.nan)
    return mass, p, {
        'mass': mass,
        'entropy': mask(h / math.log(n)) if n > 1 else np.full(mass.shape, np.nan),
        'top10': mask(csum[..., math.ceil(n * .1) - 1]),
        'top1': mask(ordered[..., 0]),
        'n90': mask(n90 / n),
    }


def tv(a, b):
    valid = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    return np.where(valid, np.nansum(np.abs(a - b), axis=-1) / 2, np.nan)


def js(a, b):
    valid = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    m = np.nan_to_num((a + b) / 2)
    return np.where(valid, entropy(m) - (entropy(np.nan_to_num(a)) + entropy(np.nan_to_num(b))) / 2, np.nan)


def top_overlap(a, b, k):
    """Intersection/k, stable index tiebreak. TV is the tie-insensitive companion."""
    valid = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    ia = np.argsort(-np.nan_to_num(a), axis=-1, kind='stable')[..., :k]
    ib = np.argsort(-np.nan_to_num(b), axis=-1, kind='stable')[..., :k]
    overlap = (ia[..., :, None] == ib[..., None, :]).any(-1).sum(-1) / k
    return np.where(valid, overlap, np.nan)


def split_clusters(items, seed='edival_extended_20261010_v1'):
    clusters = {x['session_id']: x['input_hashes'][0] for x in items}
    ordered = sorted(set(clusters.values()), key=lambda h: hashlib.sha256((seed + h).encode()).hexdigest())
    discovery = set(ordered[:len(ordered)//2])
    return {sid: {'cluster': h, 'split': 'exploration' if h in discovery else 'confirmation'} for sid, h in sorted(clusters.items())}


def analyze_turn(z):
    names = list(z['group_names']); stat = list(z['stat_names'])
    if str(z['protocol']) != 'target-token-region-stats-v2':
        raise ValueError('Only accepted v2 observations are supported')
    gs = z['group_stats'].astype(np.float64)
    if gs.shape[:3] != SHAPE or not np.array_equal(z['layer_ids'], np.arange(36)):
        raise ValueError('Unexpected axes')
    if not np.all(np.diff(z['timesteps']) < 0):
        raise ValueError('Unexpected denoising order')
    mass = gs[..., stat.index('mean')]
    if np.any(~np.isfinite(gs)) or np.any(gs < 0) or np.max(np.abs(mass.sum(-1) - 1)) > 1e-3:
        raise ValueError('Invalid probability observations')
    count = z['group_token_counts']; total_n = int(count.sum())
    values = {}; weights = {}; profiles = {}; supports = {}; metadata = {}
    turn = int(np.max(z['text_token_turns']))

    def add(key, a, weight=None):
        a = np.asarray(a, dtype=np.float64)
        if a.shape != SHAPE:
            raise ValueError((key, a.shape))
        values[key] = a
        if weight is not None:
            weights[key] = np.asarray(weight, dtype=np.float64)

    def group(g):
        return mass[..., names.index(g)]

    for i, g in enumerate(names):
        m = mass[..., i]; sd = gs[..., i, stat.index('std')]
        add(g + ':mass', m)
        add(g + ':density', m / count[i])
        add(g + ':enrichment', m / (count[i] / total_n))
        add(g + ':query_R', ratio(m*m, m*m + sd*sd), m)
        for s in ['p10', 'p50', 'p90']:
            add(g + ':query_' + s, gs[..., i, stat.index(s)])
        add(g + ':query_R_sensitivity', np.where(m > LOW, values[g + ':query_R'], np.nan), m)
        add(g + ':low_or_zero', (m <= LOW).astype(float))

    categories = {
        'current_instruction': [f'T{turn}'],
        'history_instructions': [f'T{k}' for k in range(1, turn)],
        'original_image': ['I0_vit', 'I0_vae'],
        'history_images': [f'I{k}_{mod}' for k in range(1, turn) for mod in ['vit', 'vae']],
        'target_image': ['target_image'], 'context_other': ['context_other'],
        'generation_markers': ['generation_markers'],
    }
    for cat, nn in categories.items():
        m = sum((group(g) for g in nn), np.zeros(SHAPE))
        add('category_' + cat + ':mass', m)
        nn_count = sum(count[names.index(g)] for g in nn)
        if nn_count:
            add('category_' + cat + ':density', m / nn_count)

    text_turns = z['text_token_turns']
    text = z['text_mean'].astype(np.float64)
    images = z['region_mean'].astype(np.float64).reshape(*SHAPE, turn, 2, 64)
    markers = z['image_marker_mean'].astype(np.float64)
    if markers.shape != (*SHAPE, turn, 2, 2):
        raise ValueError('Missing independent image markers')
    for t in range(1, turn+1):
        select = text_turns == t
        supports[f'T{t}'] = text[..., select]
        metadata[f'T{t}'] = {'ids': z['text_token_ids'][select].tolist(), 'offsets': z['text_token_offsets'][select].tolist()}
    for im in range(turn):
        for mi, mod in enumerate(['vit', 'vae']):
            g = f'I{im}_{mod}'
            if not np.all(z['region_token_counts'][im, mi] == (9 if mi == 0 else 16)):
                raise ValueError('Spatial token count mismatch')
            supports[g] = images[..., im, mi, :]
            spatial = group(g); block = spatial + markers[..., im, mi, :].sum(-1)
            for mk, label in enumerate(['start', 'end']):
                add(g + ':marker_' + label + '_mass', markers[..., im, mi, mk])
                add(g + ':marker_' + label + '_share', ratio(markers[..., im, mi, mk], block), block)
            add(f'I{im}_combined:mass', group(f'I{im}_vit') + group(f'I{im}_vae'))

    conditional = {}; support_mass = {}
    for g, x in supports.items():
        m, p, metrics = concentration(x)
        if np.max(np.abs(m - group(g))) > 1e-3:
            raise ValueError(('Support/group mismatch', g))
        conditional[g] = p; support_mass[g] = m
        for metric, a in metrics.items():
            if metric != 'mass':
                add(g + ':' + metric, a, m)
                add(g + ':' + metric + '_sensitivity', np.where(m > LOW, a, np.nan), m)
        # Distinct heads can focus on different elements despite equal entropy.
        avg_p = mean_valid(p, axis=2)
        head_h = mean_valid(np.where(m > 0, entropy(np.nan_to_num(p)), np.nan), axis=2)
        dispersion = entropy(np.nan_to_num(avg_p)) - head_h
        profiles[g + ':head_js'] = dispersion
        if g.startswith('I'):
            spatial = np.stack([mean_valid(p, axis=(1,2)), ratio(x.sum(axis=(1,2)), m.sum(axis=(1,2))[...,None])])
            profiles[g + ':space_step'] = spatial
            profiles[g + ':space_layer'] = np.stack([mean_valid(p, axis=(0,2)), ratio(x.sum(axis=(0,2)), m.sum(axis=(0,2))[...,None])])
            profiles[g + ':space_overall'] = np.stack([mean_valid(p, axis=(0,1,2)), ratio(x.sum(axis=(0,1,2)), m.sum())])
            add(g + ':left_share', p[...,0], m)
            add(g + ':left_mass', x[...,0])
            highest = np.where(m > 0, (np.nan_to_num(p[...,0]) >= np.nanmax(np.nan_to_num(p),axis=-1) - 1e-12).astype(float), np.nan)
            add(g + ':left_maximum', highest, m)
            edge = np.zeros((8,8),bool); edge[[0,-1],:]=True;edge[:,[0,-1]]=True
            add(g + ':edge_share', np.where(m>0,np.nansum(p[...,edge.ravel()],axis=-1),np.nan), m)
            _, _, rest = concentration(x[...,1:])
            add(g + ':entropy_without_left', rest['entropy'], x[...,1:].sum(-1))
            profiles[g + ':step_motion_TV'] = mean_valid(tv(p[1:],p[:-1]),axis=2)
            profiles[g + ':step_motion_top7'] = mean_valid(top_overlap(p[1:],p[:-1],7),axis=2)
        else:
            profiles[g + ':token_overall'] = np.stack([mean_valid(p, axis=(0,1,2)), ratio(x.sum(axis=(0,1,2)), m.sum())])
            profiles[g + ':token_head'] = mean_valid(p,axis=0)
        for k in [1,4]:
            ordered = np.sort(m,axis=2)[...,::-1]
            profiles[g + f':top{k}_head_share'] = ratio(ordered[...,:k].sum(-1),ordered.sum(-1))
        profiles[g + ':top_head_identity'] = np.where(m.sum(2)>0, m.argmax(2), -1).astype(np.int16)

    for typ, idxs in [('text', list(range(1,turn+1))), ('image', list(range(turn)))]:
        for ia, a in enumerate(idxs):
            for b in idxs[ia+1:]:
                for mod in (['text'] if typ=='text' else ['vit','vae','combined']):
                    ga,gb = (f'T{a}',f'T{b}') if typ=='text' else (f'I{a}_{mod}',f'I{b}_{mod}')
                    aa = group(ga) if mod!='combined' else values[ga+':mass']
                    bb = group(gb) if mod!='combined' else values[gb+':mass']
                    pair = aa+bb; prefix=f'pair_{ga}_{gb}'
                    r = ratio(aa,pair)
                    add(prefix+':older_share',r,pair)
                    add(prefix+':older_share_sensitivity',np.where(pair>LOW,r,np.nan),pair)
                    ca = count[names.index(ga)] if mod!='combined' else 1600
                    cb = count[names.index(gb)] if mod!='combined' else 1600
                    add(prefix+':density_share',ratio(aa/ca,aa/ca+bb/cb),pair)
                    add(prefix+':mass',pair)

    for im in range(turn):
        ga,gb=f'I{im}_vit',f'I{im}_vae'
        add(f'I{im}:vit_vae_TV',tv(conditional[ga],conditional[gb]), support_mass[ga]+support_mass[gb])
        add(f'I{im}:vit_vae_top7',top_overlap(conditional[ga],conditional[gb],7), support_mass[ga]+support_mass[gb])
    return values, weights, profiles, supports, metadata


def compact(values, weights, profiles):
    names = sorted(values)
    c = np.stack([values[k] for k in names],axis=-1)
    w = np.stack([weights.get(k,np.ones(SHAPE)) for k in names],axis=-1)
    valid = np.isfinite(c)
    pooled=lambda axes: ratio(np.where(valid,c*w,0).sum(axis=axes),np.where(valid,w,0).sum(axis=axes))
    result = {
        'names':np.array(names), 'process':mean_valid(c,2).astype(np.float32),
        'head':mean_valid(c,0).astype(np.float32), 'step':mean_valid(c,(1,2)).astype(np.float32),
        'scalar':mean_valid(c,(0,1,2)), 'weighted_scalar':pooled((0,1,2)),
        'weighted_process':pooled(2).astype(np.float32), 'n_valid':valid.sum((0,1,2)),
        'profile_names':np.array(sorted(profiles)),
    }
    for i,k in enumerate(sorted(profiles)):
        result[f'profile_{i}']=profiles[k].astype(np.float32)
    return result


def paired_turns(turns):
    """Same identity and aligned process positions; every object before role means."""
    result={}
    for a,b in [(1,2),(2,3),(1,3)]:
        va,wa,pa,sa,ma=turns[a];vb,wb,pb,sb,mb=turns[b]
        for g in sorted(sa.keys() & sb.keys()):
            if g.startswith('T') and ma[g]!=mb[g]:
                raise ValueError(('Text support changed',g))
            am,ap=distribution(sa[g]);bm,bp=distribution(sb[g]);prefix=f'{a}_{b}__{g}'
            values={};weights={}
            for metric in ['mass','entropy','top10','query_R']:
                key=g+':'+metric
                values[prefix+':delta_'+metric]=vb[key]-va[key]
            values[prefix+':TV']=tv(ap,bp)
            values[prefix+':JS']=js(ap,bp)
            values[prefix+':top_overlap']=top_overlap(ap,bp,math.ceil(sa[g].shape[-1]*.1))
            mask=(am>LOW)&(bm>LOW)
            for metric in ['TV','JS','top_overlap']:
                values[prefix+':'+metric+'_sensitivity']=np.where(mask,values[prefix+':'+metric],np.nan)
                weights[prefix+':'+metric]=am+bm
            result[prefix]=compact(values,weights,{})
    return result

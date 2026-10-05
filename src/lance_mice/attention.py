"""Read-only denoising attention observer; aggregated probabilities, not Q/K norms.

For indicator values V[k,g]=1 if key k belongs to group g, softmax(QK) @ V
is exactly attention mass by group. Fused SDPA avoids materializing Q x K.
The observation uses FP16 copies; original generation tensors are untouched.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

ATTENTION_VERSION = 'target-group-mass-v1'


def group_layout(trace, context_tokens, marker_tokens, target_tokens):
    names = ['context_other']
    labels = np.zeros(context_tokens + marker_tokens + target_tokens, dtype=np.int64)
    occupied = np.zeros(context_tokens, dtype=bool)
    for segment in trace:
        start, end = segment['tokens']
        # Bare prompts retain empty label records solely as CFG/cache boundaries.
        # They consume no tokens and therefore contribute no attention mass.
        if (segment['kind'] == 'text' and segment.get('role') == 'label'
                and segment.get('text') == '' and 0 <= start == end <= context_tokens):
            continue
        if not 0 <= start < end <= context_tokens or occupied[start:end].any():
            raise ValueError('Invalid or overlapping context token span')
        occupied[start:end] = True
        if segment['kind'] in ('vit', 'vae'):
            name = f"I{segment['image_index']}_{segment['kind']}"
        elif segment.get('role') in ('history', 'current'):
            name = f"T{segment['turn']}"
        else:
            continue
        if name in names:
            raise ValueError(f'Duplicate attention group: {name}')
        names.append(name)
        labels[start:end] = len(names) - 1
    names += ['generation_markers', 'target_image']
    labels[context_tokens:context_tokens + marker_tokens] = len(names) - 2
    labels[context_tokens + marker_tokens:] = len(names) - 1
    counts = np.bincount(labels, minlength=len(names))
    if min(counts) <= 0:
        raise ValueError('Empty attention group')
    return names, labels, counts


def indicator_mass(q, k, labels, group_count):
    """q/k [1, heads, tokens, dim]; positive branch, no padding/causal mask."""
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if group_count > q.shape[-1]:
        raise ValueError('More groups than indicator value channels')
    # BF16 Q/K values in the observed norm/RoPE range are represented more
    # accurately by FP16 output probabilities than by BF16 probabilities.
    q, k = q.to(torch.float16), k.to(torch.float16)
    basis = torch.zeros(k.shape[-2], q.shape[-1], dtype=q.dtype, device=q.device)
    basis.scatter_(1, labels[:, None], 1)
    values = basis[None, None].expand(1, k.shape[1], -1, -1)
    with torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        result = F.scaled_dot_product_attention(q, k, values, dropout_p=0.0,
            is_causal=False, scale=q.shape[-1] ** -0.5, enable_gqa=q.shape[1] != k.shape[1])
    return result[..., :group_count].float().mean(dim=-2).squeeze(0)


def reference_mass(q, k, labels, group_count):
    """Small FP32 reference only for sampled queries in acceptance and audits."""
    import torch
    q, k = q.float(), k.float()
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    with torch.autocast("cuda", enabled=False):
        probs = (q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5).softmax(-1)
    return torch.stack([probs[..., labels == i].sum(-1).mean(-1)
                        for i in range(group_count)], -1).squeeze(0)


class AttentionObserver:
    def __init__(self, bagel, trace, context_tokens, marker_tokens, target_tokens, steps):
        self.bagel = bagel
        self.names, self.labels, self.counts = group_layout(trace, context_tokens, marker_tokens, target_tokens)
        self.marker_tokens, self.target_tokens, self.steps = marker_tokens, target_tokens, steps
        self.layers = sorted((m.layer_idx, m.attn_noncausal_local)
            for m in bagel.modules() if hasattr(m, '_forward_gen') and hasattr(m, 'attn_noncausal_local'))
        if len(self.layers) != bagel.config.llm_config.num_hidden_layers:
            raise ValueError('Attention layer inventory differs from model config')
        self.rows, self.timesteps, self.handles, self.errors = {}, [], [], []
        self.step = -1

    def __enter__(self):
        import torch
        self.original_forward = self.bagel.forward
        def forward(*args, **kwargs):
            if args:
                raise ValueError('Observer requires the fixed keyword denoiser interface')
            self.step += 1
            self.timesteps.append(kwargs['timestep'][0].detach().clone())
            return self.original_forward(**kwargs)
        self.bagel.forward = forward
        for index, layer in self.layers:
            def observe(module, args, kwargs, index=index):
                q, k = args[:2]
                if q.ndim != 4 or q.shape[1] != self.marker_tokens + self.target_tokens:
                    raise ValueError('Unexpected denoising query layout')
                if k.shape[1] != len(self.labels):
                    raise ValueError('Positive branch key layout differs (padding or cache mismatch)')
                metadata = args[3] if len(args) > 3 else kwargs.get('attn_metadata')
                # The first branch is positive and the longest. Explicitly verify
                # it has no padding mask before interpreting all its keys.
                mask = getattr(metadata, 'attn_mask', None)
                if mask is not None:
                    positive_mask = mask[0]
                    if positive_mask.dtype == torch.bool:
                        if not bool(positive_mask.all()):
                            raise ValueError('Positive attention branch has masked keys')
                    elif not bool((positive_mask == 0).all()):
                        raise ValueError('Positive attention branch has additive mask')
                q = q[:1, self.marker_tokens:].permute(0, 2, 1, 3).detach()
                k = k[:1].permute(0, 2, 1, 3).detach()
                labels = torch.as_tensor(self.labels, device=q.device)
                key = (self.step, index)
                if key in self.rows or self.step < 0:
                    raise ValueError('Duplicate or untagged attention observation')
                self.rows[key] = indicator_mass(q, k, labels, len(self.names))
                # Real-Q/K reference at the first and last layers, first and
                # last timestep, three evenly spaced target queries, every head.
                if index in (self.layers[0][0], self.layers[-1][0]) and self.step in (0, self.steps-1):
                    sample = q[:, :, [0, self.target_tokens//2, self.target_tokens-1], :]
                    measured = indicator_mass(sample, k, labels, len(self.names))
                    reference = reference_mass(sample, k, labels, len(self.names))
                    self.errors.append((measured-reference).abs().max())
            self.handles.append(layer.register_forward_pre_hook(observe, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.bagel.forward = self.original_forward

    def save(self, path):
        import torch
        if self.step + 1 != self.steps or len(self.rows) != self.steps * len(self.layers):
            raise ValueError('Missing timestep/layer attention observations')
        masses = torch.stack([self.rows[(step, layer)] for step in range(self.steps)
                              for layer, _ in self.layers]).reshape(self.steps, len(self.layers), -1, len(self.names))
        masses = masses.cpu().numpy()
        reference_error = max(x.item() for x in self.errors)
        normalization_error = float(np.abs(masses.sum(-1) - 1).max())
        if (not np.isfinite(masses).all() or masses.min() < 0 or
                reference_error > 0.0005 or normalization_error > 0.001):
            raise ValueError(f'Attention probability validation failed: {reference_error=}, {normalization_error=}')
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        with path.open('xb') as f:
            np.savez_compressed(f, mass=masses, group_names=np.array(self.names),
                group_token_counts=self.counts, token_group_ids=self.labels,
                timesteps=torch.stack(self.timesteps).float().cpu().numpy(),
                layer_ids=np.array([x[0] for x in self.layers]), target_query_count=self.target_tokens,
                protocol=np.array(ATTENTION_VERSION))
        return {'version': ATTENTION_VERSION, 'file': path.name.removesuffix('.part'),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'branch': 'positive',
            'shape': list(masses.shape), 'axes': ['step', 'layer', 'head', 'group'],
            'groups': self.names, 'group_token_counts': self.counts.tolist(),
            'target_query_count': self.target_tokens, 'max_reference_absolute_error': reference_error,
            'max_probability_sum_error': normalization_error,
            'mean_mass': dict(zip(self.names, masses.mean((0,1,2)).tolist())),
            'method': 'Post-norm/RoPE Q/K copies; FP16 fused SDPA with group-indicator values; mean over target latent queries; full-key softmax. Generation unchanged.'}


def validate_attention(path, metadata, trace, context_tokens, expected_steps):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != metadata['sha256']:
        raise ValueError(f'Attention checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as data:
        mass = data['mass']
        if str(data['protocol']) != ATTENTION_VERSION or list(mass.shape) != metadata['shape']:
            raise ValueError('Attention format or shape mismatch')
        names, labels, counts = group_layout(trace, context_tokens,
            metadata['group_token_counts'][-2], metadata['target_query_count'])
        if (data['group_names'].tolist() != names or not np.array_equal(data['token_group_ids'], labels)
                or not np.array_equal(data['group_token_counts'], counts)):
            raise ValueError('Attention key-to-group mapping differs from actual model input')
        if (mass.shape[0] != expected_steps or not np.isfinite(mass).all()
                or mass.min() < 0 or np.abs(mass.sum(-1)-1).max() > 0.001):
            raise ValueError('Attention probability or timestep validation failed')
        if len(data['timesteps']) != expected_steps or not np.all(np.diff(data['timesteps']) < 0):
            raise ValueError('Attention denoising timestep order is invalid')
        return {'shape': list(mass.shape), 'groups': names,
                'max_probability_sum_error': float(np.abs(mass.sum(-1)-1).max())}

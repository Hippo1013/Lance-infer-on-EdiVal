"""CPU image identity and Lance's 768-area bucket geometry.

Bucket selection is adapted from ByteDance Lance (Apache-2.0),
data/video/transforms/bucket_resize.py at 4baeee086648996f6ab12e673cbe461b0b149997.
See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import hashlib
import math

from PIL import Image

ASPECTS = (21 / 9, 16 / 9, 4 / 3, 1.0, 3 / 4, 9 / 16)


def image_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    h = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode())
    h.update(rgb.tobytes())
    return h.hexdigest()


def bucket_size(width: int, height: int, resolution: int = 768) -> tuple[int, int]:
    if min(width, height, resolution) <= 0:
        raise ValueError("Image dimensions must be positive")
    buckets = []
    for ratio in ASPECTS:
        w1 = round(math.sqrt(resolution**2 * ratio) / 16) * 16
        h1 = round(w1 / ratio / 16) * 16
        h2 = round(math.sqrt(resolution**2 / ratio) / 16) * 16
        w2 = round(h2 * ratio / 16) * 16
        candidates = [(w1, h1), (w2, h2)]
        buckets.append(min(candidates, key=lambda wh: (
            abs(wh[0] / wh[1] - ratio), abs(wh[0] * wh[1] - resolution**2))))
    return min(buckets, key=lambda wh: abs(width / height - wh[0] / wh[1]))


def prepare_vae_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Deterministic center crop then bicubic resize; no stochastic crop RNG."""
    image = image.convert("RGB")
    w, h = image.size
    ratio = size[0] / size[1]
    if w / h < ratio:
        cw, ch = w, round(w / ratio)
    elif w / h > ratio:
        cw, ch = round(h * ratio), h
    else:
        cw, ch = w, h
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).resize(size, Image.Resampling.BICUBIC)

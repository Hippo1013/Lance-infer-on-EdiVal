from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .protocol import HISTORY_EXPLANATION, MODEL_REVISION, OMNI_REVISION, PROTOCOL_VERSION, digest


@dataclass(frozen=True)
class Settings:
    steps: int = 30
    timestep_shift: float = 3.5
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 1.0
    cfg_interval: tuple[float, float] = (0.4, 1.0)
    cfg_renorm_type: str = "global"
    cfg_renorm_min: float = 0.0
    resolution: int = 768
    seed: int = 42
    cache_mode: str = "none"

    def __post_init__(self):
        if self.steps < 1 or self.timestep_shift <= 0 or self.cfg_text_scale < 1:
            raise ValueError("Invalid denoising settings")
        if self.cfg_img_scale != 1.0:
            raise ValueError("v1 supports text CFG only (cfg_img_scale=1)")
        if self.resolution != 768:
            raise ValueError("v1 fixes resolution to the Lance 768-area buckets")
        if self.cache_mode not in {"none", "images", "prefix"}:
            raise ValueError("cache_mode must be none, images or prefix")
        if not 0 <= self.cfg_interval[0] < self.cfg_interval[1] <= 1:
            raise ValueError("Invalid CFG interval")
        if self.cfg_renorm_type != "global" or self.cfg_renorm_min != 0:
            raise ValueError("v1 fixes CFG renormalization to global/min=0")

    def identity(self) -> dict:
        return {"settings": asdict(self), "protocol": PROTOCOL_VERSION,
                "history_explanation_hash": digest(HISTORY_EXPLANATION),
                "omni_revision": OMNI_REVISION, "model_revision": MODEL_REVISION}

    def fingerprint(self) -> str:
        return digest(self.identity())

    @classmethod
    def from_file(cls, path: Path) -> Settings:
        values = json.loads(path.read_text(encoding="utf-8"))
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        if "cfg_interval" in values:
            values["cfg_interval"] = tuple(values["cfg_interval"])
        return cls(**values)

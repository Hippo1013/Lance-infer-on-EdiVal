"""Read only benchmark source images and original edit instructions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SMOKE_IDS = {"cm": "05447e326032ca20", "cu": "091097244653b968"}


@dataclass(frozen=True)
class Sample:
    split: str
    sample_id: str
    image: Path
    instructions: tuple[str, ...]

    @property
    def session_id(self) -> str:
        return f"{self.split}/{self.sample_id}"


def _load_mice(root: Path, selection: str = "smoke") -> list[Sample]:
    if selection not in {"smoke", "all"}:
        raise ValueError("selection must be smoke or all")
    root = root.resolve()
    samples = []
    seen = set()
    for split in ("cm", "cu"):
        metadata = root / split / "test_metadata.jsonl"
        for line_no, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            # MICE source_image paths are relative to the dataset root, not cm/cu.
            image = (root / row["image"]).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise ValueError(f"Missing or out-of-root image at {metadata}:{line_no}")
            raw = row["instruction"]
            if not isinstance(raw, list) or len(raw) != 3 or row.get("turns") != 3:
                raise ValueError(f"Expected three turns at {metadata}:{line_no}")
            if any(not isinstance(x, str) or not x.strip() for x in raw):
                raise ValueError(f"Invalid instruction at {metadata}:{line_no}")
            sample = Sample(split, image.stem, image, tuple(raw))
            if sample.session_id in seen:
                raise ValueError(f"Duplicate sample {sample.session_id}")
            seen.add(sample.session_id)
            if selection == "all" or sample.sample_id == SMOKE_IDS[split]:
                samples.append(sample)
    samples.sort(key=lambda x: (x.split, x.sample_id))
    if selection == "smoke" and len(samples) != 2:
        raise ValueError("Pinned CM/CU smoke samples are missing")
    return samples


def shard_samples(samples: list[Sample], worker: int, workers: int) -> list[Sample]:
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError("Invalid worker assignment")
    # Sorted cm then cu gives balanced full splits and distributes the two smoke cases.
    return [s for i, s in enumerate(sorted(samples, key=lambda x: (x.split, x.sample_id)))
            if i % workers == worker]


IMGEDIT_SPLITS = ("content_memory", "content_understand", "version_backtrace")
IMGEDIT_SMOKE_ID = "000038819"
DEFAULT_ROOTS = {"mice": Path("/home/chs/dataset/MICE-Bench"),
                 "imgedit": Path("/home/chs/dataset/ImgEdit-Bench-Multi-Turn/multiturn")}


def load_samples(root: Path, selection: str = "smoke", benchmark: str = "mice") -> list[Sample]:
    if selection not in {"smoke", "all"}:
        raise ValueError("selection must be smoke or all")
    if benchmark == "mice":
        return _load_mice(root, selection)
    if benchmark != "imgedit":
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    root = root.resolve()
    if not (root / IMGEDIT_SPLITS[0]).is_dir() and (root / "multiturn").is_dir():
        root = (root / "multiturn").resolve()
    samples, seen = [], set()
    for split in IMGEDIT_SPLITS:
        metadata = root / split / "annotation.json"
        # Despite the suffix, the official files are JSON Lines.
        for line_no, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = row.get("id")
            if not isinstance(identifier, str) or Path(identifier).name != identifier:
                raise ValueError(f"Invalid image id at {metadata}:{line_no}")
            image = (root / split / identifier).resolve()
            if not image.is_relative_to(root / split) or not image.is_file():
                raise ValueError(f"Missing or out-of-root image at {metadata}:{line_no}")
            keys = {k for k in row if k.startswith("turn")}
            if keys not in ({"turn1", "turn2"}, {"turn1", "turn2", "turn3"}):
                raise ValueError(f"Expected contiguous two or three turns at {metadata}:{line_no}")
            instructions = tuple(row[f"turn{i}"] for i in range(1, len(keys) + 1))
            if any(not isinstance(x, str) or not x.strip() for x in instructions):
                raise ValueError(f"Invalid instruction at {metadata}:{line_no}")
            sample = Sample(split, image.stem, image, instructions)
            if sample.session_id in seen:
                raise ValueError(f"Duplicate sample {sample.session_id}")
            seen.add(sample.session_id)
            if selection == "all" or sample.sample_id == IMGEDIT_SMOKE_ID:
                samples.append(sample)
    samples.sort(key=lambda s: (s.split, s.sample_id))
    if selection == "smoke" and len(samples) != len(IMGEDIT_SPLITS):
        raise ValueError("Pinned ImgEdit smoke samples are missing")
    return samples

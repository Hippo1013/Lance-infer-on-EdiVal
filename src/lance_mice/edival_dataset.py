"""EdiVal cumulative-prefix CSV to complete sessions, without evaluation inputs."""

from __future__ import annotations

import ast
import csv
import hashlib
import re
from pathlib import Path
from zipfile import ZipFile

from .dataset import Sample

CSV_NAME = "oai_instruction_generation_output.csv"
ZIP_NAME = "input_images_resize_512.zip"
SMOKE_IDS = ("0", "1")


def instruction_chains(root: Path) -> dict[str, tuple[str, ...]]:
    """Validate all three prefix rows; retain each instruction exactly as released."""
    metadata = root / CSV_NAME
    groups = {}
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_index", "turns", "instructions"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing EdiVal CSV columns: {metadata}")
        for line_no, row in enumerate(reader, 2):
            identifier, turn_text = row["image_index"], row["turns"]
            if not isinstance(identifier, str) or not re.fullmatch(r"0|[1-9][0-9]*", identifier):
                raise ValueError(f"Invalid image_index at {metadata}:{line_no}")
            if turn_text not in {"1", "2", "3"}:
                raise ValueError(f"Expected turns 1/2/3 at {metadata}:{line_no}")
            turn = int(turn_text)
            # These cells are Python list literals, not JSON. Never use eval().
            try:
                instructions = ast.literal_eval(row["instructions"])
            except (ValueError, SyntaxError, TypeError) as exc:
                raise ValueError(f"Invalid instructions at {metadata}:{line_no}") from exc
            if (not isinstance(instructions, list) or len(instructions) != turn
                    or any(not isinstance(x, str) or not x.strip() for x in instructions)):
                raise ValueError(f"Invalid instruction prefix at {metadata}:{line_no}")
            prefixes = groups.setdefault(identifier, {})
            if turn in prefixes:
                raise ValueError(f"Duplicate EdiVal row: {identifier}, turn {turn}")
            prefixes[turn] = tuple(instructions)
    chains = {}
    for identifier, prefixes in groups.items():
        if set(prefixes) != {1, 2, 3}:
            raise ValueError(f"Missing prefix rows for EdiVal session {identifier}")
        full = prefixes[3]
        if any(prefixes[t] != full[:t] for t in (1, 2)):
            raise ValueError(f"Conflicting instruction prefixes for EdiVal session {identifier}")
        chains[identifier] = full
    if not chains:
        raise ValueError("No EdiVal sessions")
    return chains


def load_edival(root: Path, selection: str = "smoke") -> list[Sample]:
    if selection not in {"smoke", "all"}:
        raise ValueError("selection must be smoke or all")
    root = root.resolve()
    chains = instruction_chains(root)
    archive_path = root / ZIP_NAME
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate EdiVal ZIP members")
        members = set(names)
        samples = []
        for identifier, instructions in chains.items():
            member = f"{identifier}_input_raw.jpg"
            if member not in members or archive.getinfo(member).is_dir():
                raise ValueError(f"Missing EdiVal source image: {member}")
            # CSV defines coverage. The archive also contains 34 unused images.
            if selection == "all" or identifier in SMOKE_IDS:
                samples.append(Sample("edival", identifier, archive_path, instructions, member))
    samples.sort(key=lambda s: s.sample_id)
    if selection == "smoke" and {s.sample_id for s in samples} != set(SMOKE_IDS):
        raise ValueError("Pinned EdiVal smoke samples are missing")
    return samples


def source_identity(root: Path) -> dict:
    """Bind both original files to run identity and reject changed-data resumes."""
    identity = {}
    for name in (CSV_NAME, ZIP_NAME):
        path = root.resolve() / name
        with path.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        identity[name] = {"sha256": sha256, "size": path.stat().st_size}
    return identity

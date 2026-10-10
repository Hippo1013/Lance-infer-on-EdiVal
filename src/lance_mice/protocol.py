"""Model-independent history protocol, RNG domains, and cache identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

PROTOCOL_VERSION = "lance-history-bare-v2"
CHAT_PROTOCOL_VERSION = "lance-history-chat-v1"
# Preserve original per-session noise/VAE seeds across this prompt-only change.
RNG_NAMESPACE = "lance-history-v1"
LEGACY_PROTOCOL_VERSION = "lance-history-v1"
OMNI_REVISION = "b742f86136d28423903a1213adde2a62a11067a2"
MODEL_REVISION = "7395315758865e6f56ab87ad06a88c7ac172f056"
HISTORY_EXPLANATION = ""
LEGACY_HISTORY_EXPLANATION = (
    "\n\nThis input contains a chronological image-editing session.\n"
    "Instructions marked HISTORY have already been applied.\n"
    "The image following each historical instruction is its resulting image.\n"
    "Apply CURRENT EDIT to the latest image.\n\n"
)


def digest(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def derive_seed(base: int, session_id: str, domain: str, index: int) -> int:
    """Stable across processes, GPU placement, resume, and PYTHONHASHSEED."""
    return int(digest([RNG_NAMESPACE, base, session_id, domain, index])[:16], 16) & ((1 << 63) - 1)


@dataclass(frozen=True)
class Segment:
    kind: str
    role: str
    turn: int
    text: str = ""
    image_index: int | None = None


def history_segments(instructions: list[str], *, version: str = PROTOCOL_VERSION) -> list[Segment]:
    if not instructions or any(not isinstance(x, str) or not x.strip() for x in instructions):
        raise ValueError("Expected nonempty raw instructions")
    if version not in {PROTOCOL_VERSION, CHAT_PROTOCOL_VERSION, LEGACY_PROTOCOL_VERSION}:
        raise ValueError(f"Unknown history protocol: {version}")
    legacy = version == LEGACY_PROTOCOL_VERSION
    chat = version == CHAT_PROTOCOL_VERSION
    # Empty labels remain as zero-token CFG/cache boundaries, never prompt text.
    segments = [Segment("image", "image", 0, image_index=0)]
    if legacy:
        segments.append(Segment("text", "framing", 0, LEGACY_HISTORY_EXPLANATION))
    for i, instruction in enumerate(instructions, 1):
        current = i == len(instructions)
        label = "CURRENT EDIT" if current else "HISTORY"
        segments.append(Segment("text", "label", i, f"{label} — Turn {i}\n" if legacy else ""))
        segments.append(Segment("text", "current" if current else "history", i, instruction))
        if not current:
            segments.append(Segment("text", "separator", i,
                "<|im_end|>\n<|im_start|>assistant\n" if chat else "\n\n"))
            segments.append(Segment("image", "image", i, image_index=i))
            segments.append(Segment("text", "separator", i,
                "<|im_end|>\n<|im_start|>user\n" if chat else "\n\n"))
    return segments


def prefix_length(segments: list[Segment]) -> int:
    """Cache ends at the zero-token boundary before the current instruction."""
    current_turn = next(s.turn for s in segments if s.role == "current")
    return next(i for i, s in enumerate(segments) if s.role == "label" and s.turn == current_turn)


def segment_signature(segment: Segment, image_hashes: list[str]) -> str:
    value = asdict(segment)
    if segment.kind == "image":
        value["image_hash"] = image_hashes[segment.image_index]
    return digest(value)


def reusable_prefix(old: list[str], new: list[str]) -> int:
    """Use a saved KV snapshot only when its entire prefix still matches."""
    return len(old) if len(old) <= len(new) and old == new[:len(old)] else 0


def render_user(instructions: list[str], *, version: str = PROTOCOL_VERSION) -> str:
    return "".join(f"[IMAGE_{s.image_index}]" if s.kind == "image" else s.text
                   for s in history_segments(instructions, version=version))


def protocol_manifest(instructions: list[str], *, version: str = PROTOCOL_VERSION) -> dict[str, Any]:
    segments = history_segments(instructions, version=version)
    return {
        "version": version,
        "segments": [asdict(s) for s in segments],
        "image_count": len(instructions),
        "instruction_count": len(instructions),
        "target_count": 1,
        "cfg_removed_segments": [i for i, s in enumerate(segments) if s.role == "current"],
        "user_prompt_preview": render_user(instructions, version=version),
    }

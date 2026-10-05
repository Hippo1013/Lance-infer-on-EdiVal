from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image

from lance_mice.acceptance import compare_runs, validate_run
from lance_mice.backend import history_payload
from lance_mice.dataset import SMOKE_IDS, load_samples, shard_samples
from lance_mice.images import bucket_size, image_hash, prepare_vae_image
from lance_mice.protocol import (derive_seed, history_segments, prefix_length,
                                 reusable_prefix, segment_signature, digest, render_user, protocol_manifest,
                                 PROTOCOL_VERSION)
from lance_mice.runner import main, run_session, write_json
from lance_mice.settings import Settings


class FakeBackend:
    """A CPU test double, never exposed by the actual inference CLI."""
    def __init__(self, settings):
        self.settings, self.calls = settings, []

    def edit(self, session_id, images, instructions, *, end_session=False):
        self.calls.append((session_id, [image_hash(x) for x in images], list(instructions)))
        # Later outputs depend on the actual prior generated pixels.
        color = images[-1].getpixel((0, 0))
        output = Image.new("RGB", images[-1].size, tuple((x + 17) % 256 for x in color))
        trace, token = [], 0
        for s in history_segments(instructions):
            if s.kind == "image":
                for kind in ("vit", "vae"):
                    trace.append({"kind": kind, "image_index": s.image_index, "tokens": [token, token + 4]})
                    token += 4
            else:
                trace.append({"kind": "text", "role": s.role, "turn": s.turn,
                              "text": s.text, "tokens": [token, token + len(s.text)]})
                token += len(s.text)
        return output, {"image_hashes": [image_hash(x) for x in images], "turn": len(instructions),
                        "seed": derive_seed(self.settings.seed, session_id, "noise", len(instructions)),
                        "positive_trace": trace, "negative_trace": [x for x in trace if x.get("role") != "current"]}


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lance-core-")
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "source_image").mkdir(parents=True)
        for split, sample_id in SMOKE_IDS.items():
            (self.dataset / split).mkdir()
            path = f"source_image/{sample_id}.png"
            Image.new("RGB", (32, 24), (31, 42, 53)).save(self.dataset / path)
            row = {"image": path, "instruction": ["Keep red. Add a leaf", "Make it black", "Add a leaf"],
                   "turns": 3, "formatted_instruction": ["SECRET ANSWER"] * 3,
                   "all_objects": ["SECRET OBJECT"]}
            (self.dataset / split / "test_metadata.jsonl").write_text(json.dumps(row) + "\n")
        self.samples = load_samples(self.dataset)
        self.settings = Settings()

    def tearDown(self):
        self.tmp.cleanup()

    def run_fixture(self, root, *, settings=None):
        settings = settings or self.settings
        write_json(root / "run.json", {"samples": [(s.session_id, s.instructions) for s in self.samples],
            "settings": settings.identity(), "model_files": {"fixture": True}})
        backend = FakeBackend(settings)
        for sample in self.samples:
            run_session(sample, root, settings, backend, run_id="fixture")
        return backend

    def test_repeated_current_instruction_is_distinct_from_history(self):
        segments = history_segments(list(self.samples[0].instructions))
        current = [s for s in segments if s.role == "current"]
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0].turn, 3)
        self.assertEqual(current[0].text, "Add a leaf")
        self.assertIn("Add a leaf", next(s.text for s in segments if s.role == "history"))
        self.assertEqual([s.image_index for s in segments if s.kind == "image"], [0, 1, 2])

    def test_bare_prompt_and_original_rng_namespace(self):
        instructions = ["Replace the helmet", "Make it brown", "Remove it"]
        self.assertEqual(render_user(instructions),
            "[IMAGE_0]Replace the helmet\n\n[IMAGE_1]\n\nMake it brown\n\n[IMAGE_2]\n\nRemove it")
        segments = history_segments(instructions)
        self.assertFalse(any(s.role == "framing" for s in segments))
        self.assertTrue(all(not s.text for s in segments if s.role == "label"))
        self.assertEqual(PROTOCOL_VERSION, "lance-history-bare-v2")
        self.assertEqual(protocol_manifest(instructions)["version"], PROTOCOL_VERSION)
        self.assertEqual(derive_seed(42, "cm/id", "noise", 3),
            int(digest(["lance-history-v1", 42, "cm/id", "noise", 3])[:16], 16) & ((1 << 63) - 1))

    def test_prefix_promotion_and_invalidation(self):
        text = list(self.samples[0].instructions)
        a, b = history_segments(text[:2]), history_segments(text)
        a = [segment_signature(s, ["a", "b"]) for s in a[:prefix_length(a)]]
        b = [segment_signature(s, ["a", "b", "c"]) for s in b[:prefix_length(b)]]
        self.assertEqual(reusable_prefix(a, b), len(a))
        altered = history_segments(["A different first instruction", *text[1:]])
        altered = [segment_signature(s, ["a", "b", "c"]) for s in altered[:prefix_length(altered)]]
        self.assertEqual(reusable_prefix(a, altered), 0)
        self.assertEqual(reusable_prefix(b, a), 0)

    def test_seeds_independent_by_purpose_sample_and_turn(self):
        seed = derive_seed(42, "cm/id", "noise", 2)
        self.assertEqual(seed, derive_seed(42, "cm/id", "noise", 2))
        self.assertEqual(len({seed, derive_seed(42, "cm/id", "vae", 2),
                            derive_seed(42, "cm/id", "noise", 3),
                            derive_seed(42, "cu/id", "noise", 2)}), 4)

    def test_shards_cover_whole_sessions_without_overlap(self):
        left, right = (shard_samples(self.samples, i, 2) for i in range(2))
        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 1)
        self.assertNotEqual(left[0].session_id, right[0].session_id)
        self.assertEqual({s.session_id for s in left + right}, {s.session_id for s in self.samples})

    def test_no_evaluation_answers_in_payload(self):
        images = [Image.new("RGB", (32, 32)) for _ in range(3)]
        payload = history_payload(self.samples[0].session_id, images,
                                  list(self.samples[0].instructions), self.settings)
        history = payload["extra_args"]["lance_history"]
        self.assertNotIn("SECRET", json.dumps(history))
        self.assertEqual(history["instructions"], list(self.samples[0].instructions))
        with self.assertRaises(ValueError):
            history_payload("s", images, ["one instruction"], self.settings)

    def test_dry_run_never_creates_outputs_or_imports_gpu(self):
        output = self.root / "must-not-exist"
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = main(["--dataset", str(self.dataset), "--output", str(output), "--dry-run", "--profile", "dp2"])
        self.assertEqual(result, 0)
        self.assertFalse(output.exists())
        manifest = json.loads(stream.getvalue())
        self.assertEqual((manifest["sessions"], manifest["turns"]), (2, 6))
        self.assertEqual([len(x) for x in manifest["assignments"]], [1, 1])

    def test_outputs_feed_back_and_resume_is_identical(self):
        sample = self.samples[0]
        complete, resumed = self.root / "complete", self.root / "resumed"
        baseline = FakeBackend(self.settings)
        a = run_session(sample, complete, self.settings, baseline, run_id="fixture")
        run_session(sample, resumed, self.settings, FakeBackend(self.settings), run_id="fixture", max_turns=1)
        backend = FakeBackend(self.settings)
        b = run_session(sample, resumed, self.settings, backend, run_id="fixture", resume=True)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual([x["output_hash"] for x in a], [x["output_hash"] for x in b])
        self.assertEqual(backend.calls[0][1][-1], a[0]["output_hash"])
        self.assertEqual(backend.calls[1][1][-1], a[1]["output_hash"])

    def test_resume_rejects_changed_settings_and_tampered_images(self):
        sample, output = self.samples[0], self.root / "output"
        run_session(sample, output, self.settings, FakeBackend(self.settings), run_id="fixture")
        with self.assertRaises(ValueError):
            run_session(sample, output, replace(self.settings, seed=43), FakeBackend(self.settings),
                        run_id="fixture", resume=True)
        Image.new("RGB", (32, 24), "white").save(output / sample.session_id / "turn_1.png")
        with self.assertRaises(ValueError):
            run_session(sample, output, self.settings, FakeBackend(self.settings), run_id="fixture", resume=True)
        # The archived source is also part of the verifiable output contract.
        with Image.open(sample.image) as original:
            original.convert("RGB").save(output / sample.session_id / "turn_1.png")
        Image.new("RGB", (32, 24), "white").save(output / sample.session_id / "turn_0_input.png")
        with self.assertRaisesRegex(ValueError, "Saved source image"):
            run_session(sample, output, self.settings, FakeBackend(self.settings), run_id="fixture", resume=True)

    def test_resume_rejects_orphan_image_and_never_overwrites(self):
        sample, output = self.samples[0], self.root / "output"
        run_session(sample, output, self.settings, FakeBackend(self.settings), run_id="fixture", max_turns=1)
        (output / sample.session_id / "turn_1.json").unlink()
        with self.assertRaises(ValueError):
            run_session(sample, output, self.settings, FakeBackend(self.settings), run_id="fixture", resume=True)

    def test_validator_checks_real_input_chain_and_cfg_removal(self):
        root = self.root / "valid"
        self.run_fixture(root)
        self.assertEqual(len(validate_run(root)), 6)
        path = root / self.samples[0].session_id / "turn_3.json"
        value = json.loads(path.read_text())
        value["backend"]["negative_trace"] = value["backend"]["positive_trace"]
        write_json(path, value)
        with self.assertRaises(ValueError):
            validate_run(root)

    def test_cache_comparison_ignores_only_cache_mode(self):
        a, b = self.root / "a", self.root / "b"
        self.run_fixture(a)
        self.run_fixture(b, settings=replace(self.settings, cache_mode="prefix"))
        self.assertEqual(compare_runs(a, b)["status"], "passed")
        value = json.loads((b / "run.json").read_text())
        value["settings"]["settings"]["steps"] = 20
        write_json(b / "run.json", value)
        with self.assertRaises(ValueError):
            compare_runs(a, b)

    def test_768_bucket_geometry_and_deterministic_crop(self):
        self.assertEqual(bucket_size(500, 500), (768, 768))
        self.assertEqual(bucket_size(1600, 900), (1024, 576))
        self.assertEqual(bucket_size(900, 1600), (576, 1024))
        for w, h in ((2100, 900), (1600, 900), (400, 300), (300, 400), (100, 100)):
            size = bucket_size(w, h)
            self.assertTrue(all(n % 16 == 0 for n in size))
            image = Image.new("RGB", (w, h), "red")
            self.assertEqual(image_hash(prepare_vae_image(image, size)), image_hash(prepare_vae_image(image, size)))

    def test_dataset_rejects_path_escape_and_missing_turn(self):
        path = self.dataset / "cm" / "test_metadata.jsonl"
        row = json.loads(path.read_text())
        row["image"] = "../../outside.png"
        path.write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            load_samples(self.dataset)
        row["image"] = f"source_image/{SMOKE_IDS['cm']}.png"
        row["instruction"] = row["instruction"][:2]
        path.write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            load_samples(self.dataset)

    def test_invalid_config_fails_before_model_load(self):
        with self.assertRaises(ValueError):
            Settings(cfg_img_scale=2.0)
        with self.assertRaises(ValueError):
            Settings(cfg_interval=(0.8, 0.2))
        path = self.root / "config.json"
        path.write_text(json.dumps({**asdict(self.settings), "unknown_parameter": 1}))
        with self.assertRaises(ValueError):
            Settings.from_file(path)


if __name__ == "__main__":
    unittest.main()

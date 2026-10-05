import unittest

from lance_mice.performance import batch_measurement, summarize_mode


class PerformanceTests(unittest.TestCase):
    def worker(self, session, finish):
        return {"sessions": [session], "started_at": 100.01, "finished_at": finish,
                "generated_turns": [[session, t] for t in (1, 2, 3)],
                "turns": [{"turn": t, "wall_seconds": 2, "seconds": {"denoise": 1.5},
                           "counts": {"vit_encodes": 1, "vae_encodes": 1},
                           "peak_memory_bytes": 2**30} for t in (1, 2, 3)]}

    def test_parallel_batch_uses_slowest_worker_and_normalizes_gpu_budget(self):
        batch = batch_measurement([self.worker("cm/a", 110), self.worker("cu/b", 115)],
                                  100, ["cm/a", "cu/b"])
        self.assertEqual(batch["batch_seconds"], 15)
        self.assertEqual(batch["sessions_per_minute"], 8)
        self.assertEqual(batch["sessions_per_gpu_minute"], 4)
        with self.assertRaises(ValueError):
            batch_measurement([self.worker("cm/a", 110)] * 2, 100, ["cm/a", "cu/b"])

    def test_summary_requires_repetitions_and_preserves_variation(self):
        batches = [batch_measurement([self.worker("cm/a", end)], 100, ["cm/a"])
                   for end in (110, 111, 150)]
        summary = summarize_mode(batches)
        self.assertEqual(summary["batch_seconds"]["median"], 11)
        self.assertEqual(summary["batch_seconds"]["max"], 50)
        self.assertEqual(summary["encoder_counts_per_batch"]["vit_encodes"], [3, 3, 3])
        with self.assertRaises(ValueError):
            summarize_mode(batches[:2])

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.paired_statistics import (
    compare_methods,
    holm_adjust,
    paired_bootstrap_ci,
    paired_sign_flip_permutation,
    paired_values,
)
from evaluation.aggregate_runs import aggregate
from evaluation.efficiency_profile import (
    detect_format,
    fallback_token_count,
    profile_run,
    profile_texts,
)


class HolmTests(unittest.TestCase):
    def test_hand_computed_example(self):
        # Standard Holm step-down on sorted p = [0.01, 0.03, 0.04]:
        #   3*0.01 = 0.03, 2*0.03 = 0.06, 1*0.04 = 0.04 -> monotone max 0.06.
        adjusted, ranks = holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
        self.assertEqual(ranks, [1, 3, 2])

    def test_paper_table_scenario(self):
        # Seven raw p = 0.0001 and one raw p = 0.0002 over a family of 8 must
        # Holm-adjust to 0.0008 for every entry, not stay unchanged.
        raw = [0.0001] * 7 + [0.0002]
        adjusted, _ = holm_adjust(raw)
        np.testing.assert_allclose(adjusted, [0.0008] * 8)

    def test_caps_at_one(self):
        adjusted, _ = holm_adjust([0.5, 0.9])
        self.assertLessEqual(max(adjusted), 1.0)


class PermutationTests(unittest.TestCase):
    def test_monte_carlo_convention(self):
        rng = np.random.default_rng(0)
        diffs = rng.normal(loc=0.5, scale=1.0, size=12)
        result = paired_sign_flip_permutation(diffs, resamples=999, seed=42)
        self.assertEqual(result["resamples"], 999)
        self.assertAlmostEqual(
            result["raw_p"], (result["exceedances"] + 1) / (999 + 1)
        )

    def test_floor_pvalue(self):
        # Strongly separated pairs: essentially no sign flip reaches the
        # observed statistic, so p sits at the (e+1)/(B+1) floor.
        diffs = np.full(16, 5.0) + np.linspace(0, 1, 16)
        result = paired_sign_flip_permutation(diffs, resamples=999, seed=7)
        self.assertEqual(result["exceedances"], 0)
        self.assertAlmostEqual(result["raw_p"], 1.0 / 1000.0)

    def test_determinism(self):
        rng = np.random.default_rng(1)
        diffs = rng.normal(size=10)
        first = paired_sign_flip_permutation(diffs, resamples=999, seed=123)
        second = paired_sign_flip_permutation(diffs, resamples=999, seed=123)
        self.assertEqual(first, second)

    def test_symmetric_data_large_p(self):
        rng = np.random.default_rng(2)
        diffs = rng.normal(size=20)  # centered at zero
        result = paired_sign_flip_permutation(diffs, resamples=999, seed=5)
        self.assertGreater(result["raw_p"], 0.2)


class BootstrapTests(unittest.TestCase):
    def test_ci_contains_observed_diff(self):
        rng = np.random.default_rng(3)
        diffs = rng.normal(loc=0.3, scale=1.0, size=25)
        lo, hi = paired_bootstrap_ci(diffs, resamples=2000, seed=11)
        observed = float(diffs.mean())
        self.assertLessEqual(lo, observed)
        self.assertLessEqual(observed, hi)
        self.assertLess(lo, hi)


class PairedValuesTests(unittest.TestCase):
    ROWS = [
        {"prediction_method": "a", "gt_id": "1", "status": "ok", "m": "1.0"},
        {"prediction_method": "a", "gt_id": "2", "status": "ok", "m": "3.0"},
        {"prediction_method": "a", "gt_id": "3", "status": "error", "m": "9.0"},
        {"prediction_method": "b", "gt_id": "1", "status": "ok", "m": "2.0"},
        {"prediction_method": "b", "gt_id": "2", "status": "ok", "m": "1.0"},
        {"prediction_method": "b", "gt_id": "3", "status": "ok", "m": "5.0"},
    ]

    def test_inner_join_on_ok_rows(self):
        gt_ids, a, b, duplicates = paired_values(self.ROWS, "a", "b", "m")
        self.assertEqual(gt_ids, ["1", "2"])
        np.testing.assert_allclose(a, [1.0, 3.0])
        np.testing.assert_allclose(b, [2.0, 1.0])
        self.assertEqual(duplicates, 0)

    def test_compare_methods_full_report(self):
        result = compare_methods(
            self.ROWS,
            "a",
            "b",
            [("m", "lower")],
            seed=2026,
            permutation_resamples=999,
            bootstrap_resamples=999,
        )
        entry = result["metrics"][0]
        self.assertEqual(entry["n_paired"], 2)
        self.assertAlmostEqual(entry["mean_diff_a_minus_b"], 0.5)
        self.assertEqual(entry["winner"], "b")  # lower is better, b is lower
        self.assertEqual(entry["holm_rank"], 1)
        self.assertEqual(result["holm_family_size"], 1)


def _write_run_csv(path: Path, method: str, values: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["prediction_method", "prediction_set", "sample_id", "gt_id", "status", "geometry.fscore"]
        )
        for index, value in enumerate(values):
            writer.writerow([method, "set", str(index), str(1000 + index), "ok", value])


class AggregateRunsTests(unittest.TestCase):
    def test_mean_and_sample_std(self):
        with tempfile.TemporaryDirectory() as tmp:
            run1 = Path(tmp) / "seed2026.csv"
            run2 = Path(tmp) / "seed2027.csv"
            _write_run_csv(run1, "twinx", [1.0, 3.0])  # run mean 2
            _write_run_csv(run2, "twinx", [2.0, 4.0])  # run mean 3
            report = aggregate(
                [run1, run2], ["seed2026", "seed2027"], ["geometry.fscore"]
            )
            entry = report["aggregate"]["twinx"]["geometry.fscore"]
            self.assertEqual(entry["n_runs"], 2)
            self.assertAlmostEqual(entry["mean"], 2.5)
            self.assertAlmostEqual(entry["std_ddof1"], np.std([2.0, 3.0], ddof=1))
            run_means = {
                row["run"]: row["mean"] for row in report["run_level"]
            }
            self.assertEqual(run_means, {"seed2026": 2.0, "seed2027": 3.0})
            digests = [item["sha256"] for item in report["audit"]["inputs"]]
            self.assertTrue(all(len(d) == 64 for d in digests))

    def test_single_run_std_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            run1 = Path(tmp) / "seed2026.csv"
            _write_run_csv(run1, "twinx", [1.0, 3.0])
            report = aggregate([run1], ["seed2026"], ["geometry.fscore"])
            entry = report["aggregate"]["twinx"]["geometry.fscore"]
            self.assertIsNone(entry["std_ddof1"])
            self.assertEqual(entry["n_runs"], 1)


class EfficiencyProfileTests(unittest.TestCase):
    RSS_SAMPLE = "rss 184 0:1 14:19 46:8"

    def test_fallback_count_on_rss(self):
        # rss, 184, 0, :, 1, 14, :, 19, 46, :, 8 -> 11 tokens.
        self.assertEqual(fallback_token_count(self.RSS_SAMPLE), 11)

    def test_detect_format(self):
        self.assertEqual(detect_format(self.RSS_SAMPLE), "relative_span")
        self.assertEqual(detect_format("vox 1,2,3 4,5,6"), "voxel")
        self.assertEqual(detect_format("no marker here"), "unknown")

    def test_profile_texts_stats(self):
        texts = [self.RSS_SAMPLE, "idx 184 198 199 200", "asp 184:1 198:19"]
        report = profile_texts(texts, ["a", "b", "c"])
        self.assertEqual(report["counter"], "fallback_regex")
        aggregate = report["aggregate"]
        self.assertEqual(aggregate["n"], 3)
        expected = [fallback_token_count(t) for t in texts]
        self.assertEqual(aggregate["max"], max(expected))
        self.assertEqual(aggregate["total"], sum(expected))
        self.assertIn("relative_span", report["per_format"])
        self.assertEqual(aggregate["mean"], np.mean(expected))

    def test_profile_run_wall_time_cpu_only(self):
        with profile_run("sleep") as profile:
            pass
        self.assertIsNotNone(profile.wall_time_s)
        self.assertGreaterEqual(profile.wall_time_s, 0.0)
        # torch is not installed in this environment; memory must degrade to None.
        self.assertIsNone(profile.peak_cuda_memory_bytes)
        self.assertFalse(profile.cuda_available)


if __name__ == "__main__":
    unittest.main()

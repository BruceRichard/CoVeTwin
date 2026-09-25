"""Reproducibility-audit regression tests.

Covers the fixes for the REPRODUCIBILITY.md implementation audit:
object-disjoint split enforcement in training/build_dataset.py and the
paper-defined normalized motion-range error in evaluation/evaluate_metrics.py.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.evaluate_metrics import (
    Alignment,
    Normalization,
    articulation_metrics,
    axis_angle,
    dof_metrics,
)
from training.build_dataset import (
    PROJECT_ROOT,
    generate_split,
    load_split_ids,
    select_object_ids,
    shuffle_records,
)


IDENTITY_ALIGNMENT = Alignment(np.eye(3), np.zeros(3), 0.0)
UNIT_NORM = Normalization(center=np.zeros(3), scale=1.0)


def _dof(joint_type, axis, limits):
    return {
        "type": joint_type,
        "slot": "C" if joint_type == "revolute" else "B",
        "axis": axis,
        "origin": [0.0, 0.0, 0.0],
        "limits": limits,
    }


class MotionRangeErrorTests(unittest.TestCase):
    def test_revolute_matches_paper_formula(self):
        # Limits are stored normalized by pi: spans are 0.5*pi and 0.25*pi rad.
        pred = _dof("revolute", [1, 0, 0], [0.0, 0.5])
        gt = _dof("revolute", [1, 0, 0], [0.0, 0.25])
        record = dof_metrics(pred, gt, UNIT_NORM, UNIT_NORM, IDENTITY_ALIGNMENT, 1.0, 1.0)
        self.assertAlmostEqual(record["revolute_range_error_rad"], 0.25 * math.pi)
        self.assertAlmostEqual(record["motion_range_error"], 0.25)
        self.assertEqual(record["motion_range_unit"], "rad")
        self.assertTrue(record["motion_range_normalized"])

    def test_revolute_paper_error_is_axis_independent(self):
        pred = _dof("revolute", [0, 1, 0], [0.0, 0.5])
        gt = _dof("revolute", [1, 0, 0], [0.0, 0.25])
        record = dof_metrics(pred, gt, UNIT_NORM, UNIT_NORM, IDENTITY_ALIGNMENT, 1.0, 1.0)
        self.assertAlmostEqual(record["motion_range_error"], 0.25)
        legacy = dof_metrics(
            pred, gt, UNIT_NORM, UNIT_NORM, IDENTITY_ALIGNMENT, 1.0, 1.0,
            legacy_range_error=True,
        )
        expected_legacy = math.pi * math.sqrt(0.5**2 + 0.25**2)
        self.assertAlmostEqual(legacy["motion_range_error"], expected_legacy)
        self.assertFalse(legacy["motion_range_normalized"])

    def test_prismatic_matches_paper_formula(self):
        # pred span 0.2 m, gt span 0.1 m, annotated GT max dimension D_o = 0.5 m.
        pred = _dof("prismatic", [1, 0, 0], [0.0, 0.2])
        gt = _dof("prismatic", [1, 0, 0], [0.0, 0.2])
        record = dof_metrics(pred, gt, UNIT_NORM, UNIT_NORM, IDENTITY_ALIGNMENT, 1.0, 0.5)
        self.assertAlmostEqual(record["prismatic_range_error_m"], 0.1)
        self.assertAlmostEqual(record["motion_range_error"], 0.2)
        self.assertEqual(record["motion_range_unit"], "m")

    def test_continuous_joints_are_excluded(self):
        pred = _dof("revolute", [1, 0, 0], [-1.0, 1.0])
        gt = _dof("revolute", [1, 0, 0], [-1.0, 1.0])
        record = dof_metrics(pred, gt, UNIT_NORM, UNIT_NORM, IDENTITY_ALIGNMENT, 1.0, 1.0)
        self.assertNotIn("motion_range_error", record)
        self.assertEqual(record["continuous_accuracy"], 1.0)

    def test_mixed_continuous_gt_excluded(self):
        pred = _dof("revolute", [1, 0, 0], [0.0, 0.5])
        gt = _dof("revolute", [1, 0, 0], [-1.0, 1.0])
        record = dof_metrics(pred, gt, UNIT_NORM, UNIT_NORM, IDENTITY_ALIGNMENT, 1.0, 1.0)
        self.assertNotIn("motion_range_error", record)
        self.assertEqual(record["continuous_accuracy"], 0.0)


class AxisErrorTests(unittest.TestCase):
    def test_axis_error_is_arccos_abs_dot(self):
        self.assertAlmostEqual(axis_angle(np.array([1, 0, 0]), np.array([1, 0, 0])), 0.0)
        # Antiparallel axes describe the same joint axis.
        self.assertAlmostEqual(axis_angle(np.array([1, 0, 0]), np.array([-1, 0, 0])), 0.0)
        self.assertAlmostEqual(
            axis_angle(np.array([1, 0, 0]), np.array([0, 1, 0])), math.pi / 2
        )
        record = dof_metrics(
            _dof("revolute", [0, 1, 0], [0.0, 0.5]),
            _dof("revolute", [1, 0, 0], [0.0, 0.5]),
            UNIT_NORM,
            UNIT_NORM,
            IDENTITY_ALIGNMENT,
            1.0,
            1.0,
        )
        self.assertAlmostEqual(record["axis_error_deg"], 90.0)


class JointTypeAccuracyTests(unittest.TestCase):
    def test_accuracy_is_correct_over_max_pred_gt(self):
        params = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        pred_data = {"group_info": {"1": [[0], "0", params, "C"]}}
        gt_data = {
            "group_info": {
                "1": [[0], "0", params, "C"],
                "2": [[1], "0", params, "B"],
            }
        }
        result = articulation_metrics(
            pred_data,
            gt_data,
            {0: 0, 1: 1},
            UNIT_NORM,
            UNIT_NORM,
            IDENTITY_ALIGNMENT,
            {"pred_max_dimension_m": 1.0, "gt_max_dimension_m": 1.0},
        )
        # One correct match out of max(1 pred, 2 gt) groups.
        self.assertAlmostEqual(result["joint_type_accuracy"], 0.5)
        self.assertEqual(result["joint_type_correct"], 1)
        self.assertEqual(result["joint_type_denominator"], 2)

    def test_type_mismatch_counts_incorrect(self):
        params = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        pred_data = {"group_info": {"1": [[0], "0", params, "B"]}}
        gt_data = {"group_info": {"1": [[0], "0", params, "C"]}}
        result = articulation_metrics(
            pred_data,
            gt_data,
            {0: 0},
            UNIT_NORM,
            UNIT_NORM,
            IDENTITY_ALIGNMENT,
            {"pred_max_dimension_m": 1.0, "gt_max_dimension_m": 1.0},
        )
        self.assertAlmostEqual(result["joint_type_accuracy"], 0.0)


class SplitEnforcementTests(unittest.TestCase):
    SPLIT_IDS = {"train": {"a", "b"}, "test": {"c"}}

    def test_train_split_excludes_test_and_unknown_objects(self):
        selected, info = select_object_ids(
            ["a", "b", "c", "d"], [], self.SPLIT_IDS, "train", allow_test_leak=False
        )
        self.assertEqual(selected, ["a", "b"])
        self.assertEqual(info["excluded_test_objects"], 1)
        self.assertEqual(info["excluded_unknown_objects"], 1)

    def test_test_split_selects_only_test_objects(self):
        selected, info = select_object_ids(
            ["a", "b", "c", "d"], [], self.SPLIT_IDS, "test", allow_test_leak=False
        )
        self.assertEqual(selected, ["c"])

    def test_allow_test_leak_includes_everything(self):
        selected, info = select_object_ids(
            ["a", "b", "c", "d"], [], None, "all", allow_test_leak=True
        )
        self.assertEqual(selected, ["a", "b", "c", "d"])
        self.assertEqual(info["excluded_test_objects"], 0)

    def test_only_filter_composes_with_split(self):
        selected, _ = select_object_ids(
            ["a", "b", "c"], ["b", "c"], self.SPLIT_IDS, "train", allow_test_leak=False
        )
        self.assertEqual(selected, ["b"])

    def test_load_split_ids_from_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps({"train": ["a"], "test": ["c", "d"]}))
            split = load_split_ids(Path(directory), path)
        self.assertEqual(split, {"train": {"a"}, "test": {"c", "d"}})

    def test_canonical_split_files_are_disjoint_1636_388(self):
        split = load_split_ids(PROJECT_ROOT / "dataset" / "splits")
        self.assertEqual(len(split["train"]), 1636)
        self.assertEqual(len(split["test"]), 388)
        self.assertFalse(split["train"] & split["test"])

    def test_generate_split_is_deterministic_and_disjoint(self):
        ids = [str(i) for i in range(2024)]
        first = generate_split(ids, test_count=388, seed=42)
        second = generate_split(ids, test_count=388, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first["test"]), 388)
        self.assertEqual(len(first["train"]), 2024 - 388)
        self.assertFalse(set(first["train"]) & set(first["test"]))

    def test_shuffle_records_is_seeded(self):
        records = [{"id": i} for i in range(50)]
        self.assertEqual(
            [r["id"] for r in shuffle_records(records, 7)],
            [r["id"] for r in shuffle_records(records, 7)],
        )
        self.assertEqual(
            sorted(r["id"] for r in shuffle_records(records, 7)), list(range(50))
        )
        self.assertEqual(
            [r["id"] for r in shuffle_records(records, None)], list(range(50))
        )


if __name__ == "__main__":
    unittest.main()

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from covetwin import inference as covetwin_inference
from covetwin.geometry_codec import (
    decode_relative_shape_spans,
    encode_relative_shape_spans,
    serialize_relative_shape_spans,
    unflatten_indices,
)
from covetwin.verification import (
    FALLBACK_EMPTY_SENTINEL,
    FALLBACK_MOST_OCCUPIED,
    FALLBACK_NO_CANDIDATES,
    REASON_EMPTY_VOXEL_SET,
    REASON_NON_MONOTONIC_SPANS,
    REASON_NON_POSITIVE_LENGTH,
    REASON_OUT_OF_GRID,
    REASON_OVERLAPPING_SPANS,
    REASON_PARSE_FAILURE,
    SELECTION_RULES,
    evaluate_candidate,
    quality_score,
    select_best_candidate,
    select_candidate,
)

FRAGMENTED = "rss 0 0:1 10:1"  # two single-voxel components
CONNECTED = "rss 0 0:3"        # one 3-voxel component


class CodecRoundTripTests(unittest.TestCase):
    def test_round_trip_still_exact(self):
        indices = np.concatenate((np.array([184]), np.arange(198, 217), np.arange(230, 238)))
        voxels = unflatten_indices(indices)
        text = serialize_relative_shape_spans(encode_relative_shape_spans(voxels))
        self.assertEqual(text, "rss 184 0:1 14:19 46:8")
        np.testing.assert_array_equal(decode_relative_shape_spans(text), voxels)


class ValidityRuleTests(unittest.TestCase):
    def _reason(self, candidate):
        evaluation, voxels = evaluate_candidate(candidate)
        self.assertFalse(evaluation.valid)
        self.assertIsNone(voxels)
        return evaluation.reason

    def test_parse_failure(self):
        self.assertEqual(self._reason("totally invalid text"), REASON_PARSE_FAILURE)
        self.assertEqual(self._reason(""), REASON_PARSE_FAILURE)
        # Loose checks pass but the strict codec rejects the stray token.
        self.assertEqual(self._reason("rss 0 0:1 stray99"), REASON_PARSE_FAILURE)

    def test_empty_voxel_set(self):
        self.assertEqual(self._reason("rss 7"), REASON_EMPTY_VOXEL_SET)
        self.assertEqual(
            self._reason(np.empty((0, 3), dtype=np.int64)), REASON_EMPTY_VOXEL_SET
        )

    def test_non_positive_length(self):
        self.assertEqual(self._reason("rss 0 0:0"), REASON_NON_POSITIVE_LENGTH)

    def test_non_monotonic_span_order(self):
        self.assertEqual(self._reason("rss 10 0:1 5:1 2:1"), REASON_NON_MONOTONIC_SPANS)

    def test_overlapping_or_duplicate_spans(self):
        self.assertEqual(self._reason("rss 0 0:3 2:2"), REASON_OVERLAPPING_SPANS)
        self.assertEqual(
            self._reason(np.array([[0, 0, 0], [0, 0, 0]])),
            REASON_OVERLAPPING_SPANS,
        )

    def test_out_of_grid_indices(self):
        self.assertEqual(self._reason("rss 32760 0:10"), REASON_OUT_OF_GRID)
        self.assertEqual(self._reason(np.array([[32, 0, 0]])), REASON_OUT_OF_GRID)

    def test_valid_candidate_has_no_reason(self):
        evaluation, voxels = evaluate_candidate(CONNECTED)
        self.assertTrue(evaluation.valid)
        self.assertIsNone(evaluation.reason)
        self.assertEqual(len(voxels), 3)


class SelectionRuleTests(unittest.TestCase):
    def test_unknown_rule_rejected(self):
        with self.assertRaises(ValueError):
            select_candidate([CONNECTED], selection_rule="bogus")

    def test_connectivity_lexicographic_ranking(self):
        selection = select_candidate(
            [FRAGMENTED, "invalid", CONNECTED], selection_rule="connectivity"
        )
        self.assertEqual(selection.selected_index, 2)
        self.assertFalse(selection.fallback)
        self.assertIsNone(selection.fallback_reason)
        np.testing.assert_array_equal(
            selection.voxels, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
        )
        # Exact ties keep the earliest candidate.
        tie = select_candidate([CONNECTED, CONNECTED], selection_rule="connectivity")
        self.assertEqual(tie.selected_index, 0)

    @staticmethod
    def _divergent_candidates():
        # A: rho=0.5, c=4, n=8  -> Q = 50 - 8 + 8/32768  ~= 42.0002
        # B: rho=0.49, c=3, n=100 -> Q = 49 - 6 + 100/32768 ~= 43.0031
        # Lexicographic (rho first) must pick A; the legacy weighted Q picks B.
        a = np.array([(0, 0, z) for z in (0, 1, 2, 3, 6, 7, 9, 11)], dtype=np.int64)
        b = np.array(
            [(1, y, z) for y in range(7) for z in range(7)]
            + [(3, y, z) for y in range(5) for z in range(5)]
            + [(3, 5, 0)]
            + [(5, y, z) for y in range(5) for z in range(5)],
            dtype=np.int64,
        )
        return a, b

    def test_connectivity_is_lexicographic_not_weighted(self):
        a, b = self._divergent_candidates()
        # Lock the scenario: A has strictly higher rho but more components,
        # while B has the higher weighted Q score.
        score_a, comps_a, _, ratio_a = quality_score(a)
        score_b, comps_b, _, ratio_b = quality_score(b)
        self.assertEqual((comps_a, comps_b), (4, 3))
        self.assertAlmostEqual(ratio_a, 0.5)
        self.assertAlmostEqual(ratio_b, 0.49)
        self.assertGreater(score_b, score_a)
        selection = select_candidate([b, a], selection_rule="connectivity")
        self.assertEqual(selection.selected_index, 1)
        np.testing.assert_array_equal(selection.voxels, a)

    def test_connectivity_weighted_is_legacy(self):
        a, b = self._divergent_candidates()
        selection = select_candidate([b, a], selection_rule="connectivity_weighted")
        self.assertEqual(selection.selected_index, 0)
        np.testing.assert_array_equal(selection.voxels, b)
        # Matches the strict legacy wrapper's ranking.
        legacy = select_best_candidate([b, a])
        self.assertEqual(legacy.selected_index, 0)
        # Exact ties keep the earliest candidate.
        tie = select_candidate(
            [CONNECTED, CONNECTED], selection_rule="connectivity_weighted"
        )
        self.assertEqual(tie.selected_index, 0)

    def test_first_selects_candidate_zero_even_if_worse(self):
        selection = select_candidate([FRAGMENTED, CONNECTED], selection_rule="first")
        self.assertEqual(selection.selected_index, 0)
        self.assertFalse(selection.fallback)

    def test_first_falls_back_when_candidate_zero_undecodable(self):
        selection = select_candidate(
            ["invalid", CONNECTED, "rss 0 0:2 5:2"], selection_rule="first"
        )
        self.assertTrue(selection.fallback)
        self.assertEqual(selection.fallback_reason, FALLBACK_MOST_OCCUPIED)
        self.assertEqual(selection.selected_index, 2)
        self.assertEqual(len(selection.voxels), 4)

    def test_first_valid(self):
        selection = select_candidate(
            ["invalid", FRAGMENTED, CONNECTED], selection_rule="first_valid"
        )
        self.assertEqual(selection.selected_index, 1)
        self.assertFalse(selection.fallback)

    def test_random_valid_is_seeded_and_uniform_over_valid(self):
        candidates = ["invalid", "rss 0 0:1", "rss 5 0:1", "rss 9 0:1"]
        first = select_candidate(candidates, selection_rule="random_valid", seed=123)
        second = select_candidate(candidates, selection_rule="random_valid", seed=123)
        self.assertEqual(first.selected_index, second.selected_index)
        self.assertIn(first.selected_index, (1, 2, 3))
        self.assertEqual(first.seed, 123)
        self.assertFalse(first.fallback)

    def test_likelihood_prefers_highest_scored_valid_candidate(self):
        candidates = ["rss 0 0:1", "invalid", "rss 5 0:1"]
        scores = [-2.0, 100.0, -1.0]  # invalid candidate must be ignored
        selection = select_candidate(
            candidates, selection_rule="likelihood", scores=scores
        )
        self.assertEqual(selection.selected_index, 2)
        self.assertEqual(selection.selected.logprob, -1.0)
        self.assertEqual(selection.evaluations[1].logprob, 100.0)

    def test_likelihood_without_scores_falls_back_to_first_valid(self):
        selection = select_candidate(
            ["invalid", FRAGMENTED, CONNECTED], selection_rule="likelihood"
        )
        self.assertEqual(selection.selected_index, 1)
        self.assertFalse(selection.fallback)

    def test_all_rules_listed(self):
        self.assertEqual(
            SELECTION_RULES,
            (
                "connectivity",
                "connectivity_weighted",
                "first",
                "first_valid",
                "random_valid",
                "likelihood",
            ),
        )


class AllInvalidFallbackTests(unittest.TestCase):
    def test_parseable_but_invalid_picks_most_occupied(self):
        # index 0: overlapping spans, salvages {0,1,2} (3 voxels)
        # index 1: non-monotonic offsets, salvages {100} (1 voxel)
        selection = select_candidate(
            ["rss 0 0:2 1:2", "rss 100 0:1 0:1"], selection_rule="connectivity"
        )
        self.assertTrue(selection.fallback)
        self.assertEqual(selection.fallback_reason, FALLBACK_MOST_OCCUPIED)
        self.assertEqual(selection.selected_index, 0)
        self.assertEqual(len(selection.voxels), 3)
        self.assertIsNotNone(selection.selected)
        self.assertFalse(selection.selected.valid)

    def test_all_unparseable_returns_empty_sentinel(self):
        selection = select_candidate(["garbage", ""], selection_rule="connectivity")
        self.assertTrue(selection.fallback)
        self.assertEqual(selection.fallback_reason, FALLBACK_EMPTY_SENTINEL)
        self.assertEqual(selection.selected_index, -1)
        self.assertIsNone(selection.selected)
        self.assertEqual(selection.voxels.shape, (0, 3))
        self.assertIsNone(selection.to_dict()["selected_score"])

    def test_no_candidates_returns_sentinel(self):
        selection = select_candidate([], selection_rule="connectivity")
        self.assertTrue(selection.fallback)
        self.assertEqual(selection.fallback_reason, FALLBACK_NO_CANDIDATES)
        self.assertEqual(selection.selected_index, -1)


class VerificationReportSchemaTests(unittest.TestCase):
    def _args(self, root: Path, **overrides) -> argparse.Namespace:
        values = dict(
            output_path=root / "predictions",
            force=False,
            remove_bg=False,
            seed=7,
            global_max_new_tokens=100,
            geometry_max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            grid_size=32,
            candidate_count=2,
            verify_candidates=True,
            selection_rule=None,
            selection_seed=None,
            save_part_ply=False,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def _run(self, root: Path, responses: list[str], **overrides) -> dict:
        image_path = root / "sample.png"
        Image.new("RGB", (16, 16), "white").save(image_path)
        with patch("covetwin.inference._generate_text", side_effect=responses):
            covetwin_inference.process_image(
                image_path,
                self._args(root, **overrides),
                model=object(),
                processor=object(),
                process_vision_info=object(),
                global_prompt="global prompt",
            )
        report_path = root / "predictions" / "sample" / "candidate_verification.json"
        return json.loads(report_path.read_text())

    def test_report_records_rule_seed_and_candidate_schema(self):
        report = self._run(
            Path(tempfile.mkdtemp()),
            [
                "Name: Test\nParts:\nl_0: body",
                "invalid",
                CONNECTED,
            ],
            selection_rule="first_valid",
            selection_seed=11,
        )
        self.assertEqual(report["selection_rule"], "first_valid")
        self.assertEqual(report["selection_seed"], 11)
        (part,) = report["parts"]
        for key in (
            "selected_index",
            "selected_score",
            "selection_rule",
            "fallback",
            "fallback_reason",
            "seed",
            "candidates",
            "part_index",
            "skipped",
        ):
            self.assertIn(key, part)
        self.assertEqual(part["selected_index"], 1)
        self.assertFalse(part["fallback"])
        self.assertFalse(part["skipped"])
        self.assertEqual(len(part["candidates"]), 2)
        for candidate in part["candidates"]:
            for key in (
                "index",
                "valid",
                "reason",
                "voxel_count",
                "component_count",
                "largest_component_size",
                "largest_component_ratio",
                "score",
                "error",
                "logprob",
                "file",
            ):
                self.assertIn(key, candidate)
            self.assertNotIn("raw_text", candidate)
        self.assertEqual(part["candidates"][0]["reason"], REASON_PARSE_FAILURE)
        self.assertIsNone(part["candidates"][1]["reason"])

    def test_all_unparseable_part_is_skipped_gracefully(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._run(
                root,
                ["Name: Test\nParts:\nl_0: body", "garbage", "also garbage"],
            )
            (part,) = report["parts"]
            self.assertEqual(part["selected_index"], -1)
            self.assertTrue(part["fallback"])
            self.assertEqual(part["fallback_reason"], FALLBACK_EMPTY_SENTINEL)
            self.assertTrue(part["skipped"])
            output = root / "predictions" / "sample"
            self.assertFalse((output / "ind_0.npy").exists())
            allind = np.load(output / "allind.npy")
            self.assertEqual(allind.shape, (0, 3))

    def test_no_verify_means_first_rule(self):
        report = self._run(
            Path(tempfile.mkdtemp()),
            ["Name: Test\nParts:\nl_0: body", FRAGMENTED, CONNECTED],
            verify_candidates=False,
        )
        self.assertEqual(report["selection_rule"], "first")
        self.assertEqual(report["selection_seed"], 7)  # falls back to --seed
        self.assertEqual(report["parts"][0]["selected_index"], 0)


if __name__ == "__main__":
    unittest.main()

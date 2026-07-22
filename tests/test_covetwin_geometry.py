import unittest

import numpy as np
import torch

from covetwin.geometry_codec import (
    RelativeShapeSpans,
    decode_relative_shape_spans,
    encode_relative_shape_spans,
    parse_relative_shape_spans,
    serialize_relative_shape_spans,
    unflatten_indices,
)
from covetwin.ablation_codecs import REPRESENTATIONS, decode_geometry, encode_geometry
from covetwin.verification import evaluate_candidate, quality_score, select_best_candidate
from covetwin.flow_matching import (
    covetwin_flow_matching_loss,
    interpolate_flow_path,
    target_velocity,
)


class RelativeShapeSpanTests(unittest.TestCase):
    def test_paper_equations_round_trip(self):
        indices = np.concatenate(
            (np.array([184]), np.arange(198, 217), np.arange(230, 238))
        )
        voxels = unflatten_indices(indices)
        encoded = encode_relative_shape_spans(voxels)
        self.assertEqual(encoded.base, 184)
        self.assertEqual(encoded.spans, ((0, 1), (14, 19), (46, 8)))
        self.assertEqual(
            serialize_relative_shape_spans(encoded),
            "rss 184 0:1 14:19 46:8",
        )
        np.testing.assert_array_equal(decode_relative_shape_spans(encoded), voxels)

    def test_random_round_trip_and_deduplication(self):
        rng = np.random.default_rng(17)
        voxels = rng.integers(0, 32, size=(1000, 3), dtype=np.int64)
        voxels = np.concatenate((voxels, voxels[:20]), axis=0)
        encoded = encode_relative_shape_spans(voxels)
        decoded = decode_relative_shape_spans(serialize_relative_shape_spans(encoded))
        expected = np.unique(voxels, axis=0)
        expected = expected[np.lexsort((expected[:, 2], expected[:, 1], expected[:, 0]))]
        np.testing.assert_array_equal(decoded, expected)

    def test_parser_accepts_final_answer_after_explanation(self):
        parsed = parse_relative_shape_spans(
            "The final compact representation is:\n```text\nrss 184 0:1 14:19 46:8\n```"
        )
        self.assertEqual(parsed.base, 184)
        self.assertEqual(parsed.voxel_count, 28)

    def test_rejects_empty_out_of_bounds_and_nonmaximal_spans(self):
        with self.assertRaises(ValueError):
            encode_relative_shape_spans(np.empty((0, 3), dtype=np.int64))
        with self.assertRaises(ValueError):
            encode_relative_shape_spans(np.array([[32, 0, 0]]))
        with self.assertRaises(ValueError):
            parse_relative_shape_spans("rss 10 0:2 2:1")
        with self.assertRaises(ValueError):
            RelativeShapeSpans(32767, ((0, 2),)).validate()
        with self.assertRaises(ValueError):
            unflatten_indices(np.array([1.5]))

    def test_all_ablation_serializations_round_trip(self):
        voxels = unflatten_indices(np.array([0, 1, 2, 40, 42, 43, 44, 32767]))
        for representation in REPRESENTATIONS:
            with self.subTest(representation=representation):
                text = encode_geometry(voxels, representation)
                np.testing.assert_array_equal(
                    decode_geometry(text, representation), voxels
                )


class CandidateVerificationTests(unittest.TestCase):
    def test_exact_quality_equation(self):
        coherent = np.array([[0, 0, z] for z in range(6)], dtype=np.int64)
        score, components, largest, ratio = quality_score(coherent)
        self.assertEqual(components, 1)
        self.assertEqual(largest, 6)
        self.assertEqual(ratio, 1.0)
        self.assertAlmostEqual(score, 100.0 - 2.0 + 6.0 / 32768.0)

    def test_verifier_prefers_connected_candidate(self):
        connected = np.array([[0, 0, z] for z in range(6)], dtype=np.int64)
        fragmented = np.array(
            [[0, 0, 0], [2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10]],
            dtype=np.int64,
        )
        candidates = [
            serialize_relative_shape_spans(encode_relative_shape_spans(fragmented)),
            "not a parseable candidate",
            serialize_relative_shape_spans(encode_relative_shape_spans(connected)),
        ]
        selection = select_best_candidate(candidates)
        self.assertEqual(selection.selected_index, 2)
        self.assertFalse(selection.evaluations[1].valid)
        np.testing.assert_array_equal(selection.voxels, connected)

    def test_all_invalid_candidates_raise(self):
        evaluation, voxels = evaluate_candidate("rss 7")
        self.assertFalse(evaluation.valid)
        self.assertIsNone(voxels)
        with self.assertRaisesRegex(ValueError, "all geometry candidates are invalid"):
            select_best_candidate(["", "rss 7"])

    def test_exact_ties_keep_first_candidate(self):
        candidate = "rss 0 0:3"
        selection = select_best_candidate([candidate, candidate])
        self.assertEqual(selection.selected_index, 0)


class FlowMatchingTests(unittest.TestCase):
    def test_exact_interpolation_and_velocity(self):
        x0 = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        noise = torch.tensor([[5.0, 7.0], [6.0, 8.0]])
        t = torch.tensor([0.0, 0.5])
        expected = torch.tensor([[1.0, 3.0], [4.0, 6.0]])
        torch.testing.assert_close(interpolate_flow_path(x0, t, noise), expected)
        torch.testing.assert_close(target_velocity(x0, noise), noise - x0)

    def test_zero_loss_for_exact_velocity_model(self):
        x0 = torch.tensor([[1.0, 2.0]])
        noise = torch.tensor([[3.0, 5.0]])

        class ExactModel:
            def __call__(self, xt, t, image_condition, coarse_condition):
                self.seen_coarse = coarse_condition
                return noise - x0

        model = ExactModel()
        result = covetwin_flow_matching_loss(
            model,
            x0,
            image_condition="image",
            coarse_voxel_condition="verified_voxels",
            t=torch.tensor([0.25]),
            noise=noise,
        )
        self.assertEqual(result["loss"].item(), 0.0)
        self.assertEqual(model.seen_coarse, "verified_voxels")


if __name__ == "__main__":
    unittest.main()

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from covetwin import inference as covetwin_inference


class StageOneContractTests(unittest.TestCase):
    def _args(self, root: Path, verify: bool = True) -> argparse.Namespace:
        return argparse.Namespace(
            output_path=root / "predictions",
            force=False,
            remove_bg=False,
            seed=7,
            global_max_new_tokens=100,
            geometry_max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            grid_size=32,
            candidate_count=3,
            verify_candidates=verify,
            save_part_ply=False,
        )

    def test_verified_output_keeps_legacy_stage_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            responses = [
                "Name: Test\nParts:\nl_0: body\nl_1: handle",
                "rss 0 0:1 10:1",  # fragmented
                "rss 0 0:2",       # selected
                "invalid",
                "rss 100 0:1 10:1",
                "rss 100 0:3",     # selected
                "rss 100 0:1 20:1",
            ]
            with patch("covetwin.inference._generate_text", side_effect=responses):
                result = covetwin_inference.process_image(
                    image_path,
                    self._args(root),
                    model=object(),
                    processor=object(),
                    process_vision_info=object(),
                    global_prompt="global prompt",
                )
            self.assertEqual(result["status"], "ok")
            output = root / "predictions" / "sample"
            for name in (
                "basic_info.txt",
                "coord_0.txt",
                "coord_1.txt",
                "ind_0.npy",
                "ind_1.npy",
                "allind.npy",
                "candidate_verification.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            np.testing.assert_array_equal(
                np.load(output / "ind_0.npy"), np.array([[0, 0, 0], [0, 0, 1]])
            )
            np.testing.assert_array_equal(
                np.load(output / "ind_1.npy"), np.array([[0, 3, 4], [0, 3, 5], [0, 3, 6]])
            )
            report = json.loads((output / "candidate_verification.json").read_text())
            self.assertEqual([part["selected_index"] for part in report["parts"]], [1, 1])
            self.assertTrue(report["verification_enabled"])

    def test_no_verification_forces_candidate_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            responses = [
                "Name: Test\nParts:\nl_0: body",
                "rss 0 0:1 10:1",  # lower score, but selected without verifier
                "rss 0 0:2",
                "rss 0 0:3",
            ]
            with patch("covetwin.inference._generate_text", side_effect=responses):
                covetwin_inference.process_image(
                    image_path,
                    self._args(root, verify=False),
                    model=object(),
                    processor=object(),
                    process_vision_info=object(),
                    global_prompt="global prompt",
                )
            report_path = root / "predictions" / "sample" / "candidate_verification.json"
            report = json.loads(report_path.read_text())
            self.assertEqual(report["parts"][0]["selected_index"], 0)
            self.assertFalse(report["verification_enabled"])


if __name__ == "__main__":
    unittest.main()

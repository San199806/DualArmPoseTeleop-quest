import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dual_arm_pose_teleop.geometry import project_two_link


class TwoLinkProjectionTests(unittest.TestCase):
    def test_output_preserves_configured_link_lengths(self):
        upper = 0.500528720455
        forearm = 0.430614676944
        shoulder = np.array([0.0, 0.0, 1.4])
        raw_elbow = np.array([0.4, 0.2, 1.3])
        raw_wrist = np.array([1.2, 0.2, 1.3])
        elbow, wrist, _ = project_two_link(
            shoulder, raw_elbow, raw_wrist, upper, forearm
        )
        self.assertAlmostEqual(np.linalg.norm(elbow - shoulder), upper, places=8)
        self.assertAlmostEqual(np.linalg.norm(wrist - elbow), forearm, places=8)

    def test_full_human_extension_keeps_singularity_margin(self):
        upper = 0.5
        forearm = 0.43
        shoulder = np.zeros(3)
        _, wrist, _ = project_two_link(
            shoulder,
            np.array([0.5, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            upper,
            forearm,
            maximum_reach_ratio=0.97,
        )
        self.assertAlmostEqual(np.linalg.norm(wrist), (upper + forearm) * 0.97)


if __name__ == "__main__":
    unittest.main()

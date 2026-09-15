import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dual_arm_pose_teleop.xrobot_input import XRobotInput


class XRobotInputTests(unittest.TestCase):
    def test_unity_basis_conversion(self):
        adapter = XRobotInput.__new__(XRobotInput)
        pose = adapter._convert_pose([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(pose.position, [1.0, -3.0, 2.0])
        np.testing.assert_allclose(
            pose.rotation.as_matrix(), adapter.UNITY_TO_RIGHT_HANDED.as_matrix()
        )

    def test_invalid_pose_is_rejected(self):
        adapter = XRobotInput.__new__(XRobotInput)
        with self.assertRaises(ValueError):
            adapter._convert_pose([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            adapter._convert_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

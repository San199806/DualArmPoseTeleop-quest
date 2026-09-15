import pathlib
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dual_arm_pose_teleop.mapper import PoseArmMapper
from dual_arm_pose_teleop.types import Pose, SensorFrame


class PoseMapperTests(unittest.TestCase):
    def test_calibration_maps_to_feasible_robot_targets(self):
        identity = Rotation.identity()
        frame = SensorFrame(
            head=Pose(np.array([0.0, 0.0, 1.70]), identity),
            waist=Pose(np.zeros(3), identity),
            left_elbow=Pose(np.array([0.0, 0.30, 0.10]), identity),
            right_elbow=Pose(np.array([0.0, -0.30, 0.10]), identity),
            left_controller=Pose(np.array([0.25, 0.50, 0.05]), identity),
            right_controller=Pose(np.array([0.25, -0.50, 0.05]), identity),
        )
        robot = {
            "base": Pose(np.zeros(3), identity),
            "left_shoulder": Pose(np.array([0.0, 0.20, 1.40]), identity),
            "left_elbow": Pose(np.array([0.40, 0.25, 1.20]), identity),
            "left_wrist": Pose(np.array([0.60, 0.25, 1.30]), identity),
            "right_shoulder": Pose(np.array([0.0, -0.20, 1.40]), identity),
            "right_elbow": Pose(np.array([0.40, -0.25, 1.20]), identity),
            "right_wrist": Pose(np.array([0.60, -0.25, 1.30]), identity),
        }
        mapper = PoseArmMapper(
            {
                "upper_arm_axis": "-y",
                "elbow_sensor_to_joint_m": 0.02,
                "left_controller_to_wrist_m": [0.0, 0.0, 0.0],
                "right_controller_to_wrist_m": [0.0, 0.0, 0.0],
            },
            {"maximum_reach_ratio": 0.97, "reach_margin_m": 0.01},
            {"mode": "cooperative_wrist"},
        )
        mapper.calibrate(frame, robot)
        human_shoulders = {
            side: frame.head.position
            + frame.waist.rotation.apply(
                mapper.arms[side].head_to_shoulder_body
            )
            for side in ("left", "right")
        }
        self.assertAlmostEqual(
            human_shoulders["left"][2], human_shoulders["right"][2], places=10
        )
        self.assertAlmostEqual(
            np.linalg.norm(
                human_shoulders["left"] - human_shoulders["right"]
            ),
            0.38,
            places=10,
        )
        targets = mapper.map(frame)
        unreliable_elbow_frame = SensorFrame(
            head=frame.head,
            waist=frame.waist,
            left_elbow=Pose(np.array([10.0, -5.0, 3.0]), identity),
            right_elbow=Pose(np.array([-8.0, 4.0, -2.0]), identity),
            left_controller=frame.left_controller,
            right_controller=frame.right_controller,
        )
        targets_with_bad_elbow_positions = mapper.map(unreliable_elbow_frame)
        np.testing.assert_allclose(
            targets_with_bad_elbow_positions.left.wrist.position,
            targets.left.wrist.position,
        )
        np.testing.assert_allclose(
            targets_with_bad_elbow_positions.right.wrist.position,
            targets.right.wrist.position,
        )

        body_shift = np.array([0.40, -0.25, 0.15])
        translated_frame = SensorFrame(
            head=Pose(frame.head.position + body_shift, identity),
            waist=Pose(frame.waist.position + body_shift, identity),
            left_elbow=Pose(frame.left_elbow.position + body_shift, identity),
            right_elbow=Pose(frame.right_elbow.position + body_shift, identity),
            left_controller=Pose(
                frame.left_controller.position + body_shift, identity
            ),
            right_controller=Pose(
                frame.right_controller.position + body_shift, identity
            ),
        )
        translated_targets = mapper.map(translated_frame)
        np.testing.assert_allclose(
            translated_targets.left.wrist.position, targets.left.wrist.position
        )
        np.testing.assert_allclose(
            translated_targets.right.wrist.position, targets.right.wrist.position
        )
        for side in ("left", "right"):
            calibration = mapper.arms[side]
            target = getattr(targets, side)
            shoulder = calibration.shoulder
            elbow_local = target.elbow.position
            wrist_local = target.wrist.position
            self.assertAlmostEqual(
                np.linalg.norm(elbow_local - shoulder), calibration.upper_length, places=8
            )
            self.assertAlmostEqual(
                np.linalg.norm(wrist_local - elbow_local), calibration.forearm_length, places=8
            )
            rotation_error = (
                robot[f"{side}_wrist"].rotation.inv() * target.wrist.rotation
            ).magnitude()
            self.assertAlmostEqual(rotation_error, 0.0, places=8)

        saved = mapper.export_calibration()
        self.assertEqual(saved["version"], 5)
        restored = PoseArmMapper(
            {
                "upper_arm_axis": "-y",
                "elbow_sensor_to_joint_m": 0.02,
                "left_controller_to_wrist_m": [0.0, 0.0, 0.0],
                "right_controller_to_wrist_m": [0.0, 0.0, 0.0],
            },
            {"maximum_reach_ratio": 0.97, "reach_margin_m": 0.01},
            {"mode": "cooperative_wrist"},
        )
        restored.load_calibration(saved, robot)
        restored_targets = restored.map(frame)
        for side in ("left", "right"):
            expected = getattr(targets, side)
            actual = getattr(restored_targets, side)
            np.testing.assert_allclose(actual.elbow.position, expected.elbow.position)
            np.testing.assert_allclose(actual.wrist.position, expected.wrist.position)
            self.assertAlmostEqual(
                (expected.wrist.rotation.inv() * actual.wrist.rotation).magnitude(),
                0.0,
                places=8,
            )

    def test_cooperative_mode_preserves_relative_wrist_distance(self):
        identity = Rotation.identity()
        frame = SensorFrame(
            head=Pose(np.array([0.0, 0.0, 1.70]), identity),
            waist=Pose(np.zeros(3), identity),
            left_elbow=Pose(np.array([0.0, 0.25, 1.2]), identity),
            right_elbow=Pose(np.array([0.0, -0.25, 1.2]), identity),
            left_controller=Pose(np.array([0.35, 0.35, 1.0]), identity),
            right_controller=Pose(np.array([0.35, -0.35, 1.0]), identity),
        )
        robot = {
            "base": Pose(np.zeros(3), identity),
            "left_shoulder": Pose(np.array([0.0, 0.20, 1.40]), identity),
            "left_elbow": Pose(np.array([0.30, 0.25, 1.20]), identity),
            "left_wrist": Pose(np.array([0.45, 0.25, 1.20]), identity),
            "right_shoulder": Pose(np.array([0.0, -0.20, 1.40]), identity),
            "right_elbow": Pose(np.array([0.30, -0.25, 1.20]), identity),
            "right_wrist": Pose(np.array([0.45, -0.25, 1.20]), identity),
        }
        mapper = PoseArmMapper(
            {
                "upper_arm_axis": "-y",
                "elbow_sensor_to_joint_m": 0.02,
                "head_position_filter_alpha": 1.0,
                "left_controller_to_wrist_m": [0.0, 0.0, 0.0],
                "right_controller_to_wrist_m": [0.0, 0.0, 0.0],
            },
            {"maximum_reach_ratio": 0.97, "reach_margin_m": 0.01},
            {"mode": "cooperative_wrist"},
        )
        mapper.calibrate(frame, robot)
        initial = mapper.map(frame)
        np.testing.assert_allclose(
            initial.left.wrist.position, robot["left_wrist"].position, atol=1e-8
        )
        np.testing.assert_allclose(
            initial.right.wrist.position, robot["right_wrist"].position, atol=1e-8
        )
        shift = np.array([0.0, 0.10, 0.0])
        moved = SensorFrame(
            head=frame.head,
            waist=frame.waist,
            left_elbow=frame.left_elbow,
            right_elbow=frame.right_elbow,
            left_controller=Pose(frame.left_controller.position + shift, identity),
            right_controller=Pose(frame.right_controller.position + shift, identity),
        )
        target = mapper.map(moved)
        initial_distance = np.linalg.norm(
            initial.left.wrist.position - initial.right.wrist.position
        )
        moved_distance = np.linalg.norm(
            target.left.wrist.position - target.right.wrist.position
        )
        self.assertAlmostEqual(initial_distance, moved_distance, places=6)


if __name__ == "__main__":
    unittest.main()

import pathlib
import sys
import unittest
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/"src"))
from dual_arm_pose_teleop.mapper import PoseArmMapper
from dual_arm_pose_teleop.types import Pose, SensorFrame


class CooperativeTests(unittest.TestCase):
    def setUp(self):
        def pose(xyz, rotation=None):
            return Pose(np.array(xyz, dtype=float), Rotation.identity() if rotation is None else rotation)
        self.frame = SensorFrame(
            pose([0, 0, 1.8]), pose([0, 0, 1]),
            pose([-.2, 0, 1.2]), pose([.2, 0, 1.2]),
            pose([-.2, .3, 1.2]), pose([.2, .3, 1.2]),
        )
        self.robot = {
            "base": pose([0, 0, 0]),
            "left_shoulder": pose([-.2, 0, 1.5]),
            "right_shoulder": pose([.2, 0, 1.5]),
            "left_elbow": pose([-.2, 0, 1.2]),
            "right_elbow": pose([.25, .1, 1.2]),
            "left_wrist": pose([-.2, .3, 1.2]),
            "right_wrist": pose([.2, .3, 1.2], Rotation.from_euler("x", .6)),
        }
        self.mapper = PoseArmMapper(
            {"upper_arm_axis": "z", "elbow_sensor_to_joint_m": 0,
             "head_position_filter_alpha": 1,
             "left_controller_to_wrist_m": [0,0,0], "right_controller_to_wrist_m": [0,0,0]},
            {"maximum_reach_ratio": .97, "reach_margin_m": .01},
            {"mode": "cooperative_wrist"},
        )
        self.mapper.calibrate(self.frame, self.robot)

    def test_common_world_rotation_increment_despite_different_arm_calibration(self):
        initial = self.mapper.map(self.frame)
        delta = Rotation.from_euler("z", .3)
        frame = replace(self.frame,
            left_controller=replace(self.frame.left_controller, rotation=delta),
            right_controller=replace(self.frame.right_controller, rotation=delta))
        mapped = self.mapper.map(frame)
        left_delta = mapped.left.wrist.rotation * initial.left.wrist.rotation.inv()
        right_delta = mapped.right.wrist.rotation * initial.right.wrist.rotation.inv()
        np.testing.assert_allclose(left_delta.as_matrix(), right_delta.as_matrix(), atol=1e-9)

    def test_unreachable_pair_holds_both_positions_and_attitudes(self):
        initial = self.mapper.map(self.frame)
        frame = replace(self.frame, left_controller=Pose(np.array([-10., .3, 1.2]), Rotation.from_euler("x", 1)))
        held = self.mapper.map(frame)
        self.assertIs(held, initial)
        self.assertEqual(self.mapper.last_debug_metrics["mapping_pair_held"], 1)

    def test_inner_reach_violation_is_rejected(self):
        wrists = {s: self.mapper.arms[s].shoulder.copy() for s in ("left", "right")}
        self.assertFalse(self.mapper._wrists_feasible(wrists))
        initial = {s: p.copy() for s, p in self.mapper.last_feasible_wrists.items()}
        result, _, _ = self.mapper._couple_raw_wrists(wrists)
        for s in initial:
            np.testing.assert_allclose(result[s], initial[s])

    def test_reentry_recovers_from_held_pair(self):
        self.mapper.map(replace(self.frame, left_controller=Pose(np.array([-10., 0, 0]), Rotation.identity())))
        recovered = self.mapper.map(self.frame)
        self.assertEqual(self.mapper.last_debug_metrics["mapping_pair_held"], 0)
        np.testing.assert_allclose(recovered.left.wrist.position, self.robot["left_wrist"].position)

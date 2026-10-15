from dataclasses import replace
import pathlib
import sys
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dual_arm_pose_teleop.mapper import PoseArmMapper
from dual_arm_pose_teleop.types import Pose, SensorFrame


class PoseMapperTests(unittest.TestCase):
    def test_torso_yaw_follows_the_body_and_respects_limits(self):
        commanded = PoseArmMapper.torso_joint_target(
            current_yaw=0.4,
            calibration_yaw=0.1,
            initial=0.0,
            lower=-0.5,
            upper=0.5,
            current_joint=0.0,
            max_step=0.1,
        )
        self.assertAlmostEqual(commanded, 0.1)
        limited = PoseArmMapper.torso_joint_target(
            current_yaw=2.0,
            calibration_yaw=0.0,
            initial=0.0,
            lower=-0.2,
            upper=0.2,
            current_joint=0.0,
            max_step=1.0,
        )
        self.assertAlmostEqual(limited, 0.2)
        self.assertAlmostEqual(PoseArmMapper.torso_pitch([0, 0, 0], [0, 0, 0.4], 0.0), 0.0)
        self.assertGreater(PoseArmMapper.torso_pitch([0, 0, 0], [0.2, 0, 0.4], 0.0), 0.0)
        turned = PoseArmMapper.torso_pitch([0, 0, 0], [0, 0.2, 0.4], np.pi / 2)
        self.assertGreater(turned, 0.0)
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

    def test_skeleton_bone_lengths_scale_onto_robot_links(self):
        identity = Rotation.identity()
        left_shoulder = np.array([0.0, 0.15, 1.4])
        right_shoulder = np.array([0.0, -0.15, 1.4])
        left_elbow = left_shoulder + np.array([0.0, 0.25, 0.0])
        right_elbow = right_shoulder + np.array([0.0, -0.25, 0.0])
        left_wrist = left_elbow + np.array([0.22, 0.0, 0.0])
        right_wrist = right_elbow + np.array([0.22, 0.0, 0.0])

        def arm_rotation(direction):
            y_axis = direction/np.linalg.norm(direction)
            x_axis = np.cross(y_axis, np.array([0.0, 0.0, 1.0]))
            x_axis = x_axis/np.linalg.norm(x_axis)
            z_axis = np.cross(x_axis, y_axis)
            return Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis]))

        frame = SensorFrame(
            head=Pose(0.5*(left_shoulder+right_shoulder), identity),
            waist=Pose(np.zeros(3), identity),
            left_elbow=Pose(left_elbow, arm_rotation(left_elbow-left_shoulder)),
            right_elbow=Pose(right_elbow, arm_rotation(right_elbow-right_shoulder)),
            left_controller=Pose(left_wrist, identity),
            right_controller=Pose(right_wrist, identity),
            left_shoulder=Pose(left_shoulder, identity),
            right_shoulder=Pose(right_shoulder, identity),
        )
        robot = {
            "base": Pose(np.zeros(3), identity),
            "left_shoulder": Pose(np.array([0.0, 0.20, 1.40]), identity),
            "left_elbow": Pose(np.array([0.0, 0.60, 1.40]), identity),
            "left_wrist": Pose(np.array([0.35, 0.60, 1.40]), identity),
            "right_shoulder": Pose(np.array([0.0, -0.20, 1.40]), identity),
            "right_elbow": Pose(np.array([0.0, -0.60, 1.40]), identity),
            "right_wrist": Pose(np.array([0.35, -0.60, 1.40]), identity),
        }
        mapper = PoseArmMapper(
            {
                "upper_arm_axis": "-y",
                "elbow_sensor_to_joint_m": 0.0,
                "head_position_filter_alpha": 1.0,
                "human_upper_arm_length_m": 0.30,
                "human_forearm_length_m": 0.26,
                "head_position_filter_alpha": 1.0,
                "left_controller_to_wrist_m": [0.0, 0.0, 0.0],
                "right_controller_to_wrist_m": [0.0, 0.0, 0.0],
            },
            {"maximum_reach_ratio": 0.97, "reach_margin_m": 0.01},
            {"mode": "cooperative_wrist"},
        )
        averaged = PoseArmMapper.average_frame([frame, frame])
        self.assertIsNotNone(averaged.left_shoulder)
        mapper.calibrate(averaged, robot)
        self.assertTrue(mapper.skeleton_segments)
        live = mapper._map_cooperative(
            frame,
            frame.head.position,
            {
                side: mapper._controller_wrist_world(frame, side)
                for side in ("left", "right")
            },
        )
        for side in ("left", "right"):
            np.testing.assert_allclose(
                getattr(live, side).wrist.position,
                robot[f"{side}_wrist"].position,
                atol=1e-4,
            )
        mapper.map(frame)
        self.assertTrue(mapper.at_calibration_pose)
        self.assertAlmostEqual(mapper.human_upper_lengths["left"], 0.25)
        self.assertAlmostEqual(mapper.human_forearm_lengths["left"], 0.22)
        bent_wrist = left_elbow + np.array([0.0, 0.0, -0.22])
        bent = SensorFrame(
            head=frame.head,
            waist=frame.waist,
            left_elbow=frame.left_elbow,
            right_elbow=frame.right_elbow,
            left_controller=Pose(bent_wrist, identity),
            right_controller=frame.right_controller,
            left_shoulder=frame.left_shoulder,
            right_shoulder=frame.right_shoulder,
        )
        target = mapper.map(bent).left
        self.assertFalse(mapper.at_calibration_pose)
        self.assertAlmostEqual(
            np.linalg.norm(target.elbow.position-mapper.arms["left"].shoulder),
            mapper.arms["left"].upper_length,
            places=6,
        )
        self.assertAlmostEqual(
            np.linalg.norm(target.wrist.position-target.elbow.position),
            mapper.arms["left"].forearm_length,
            places=6,
        )
        straight_wrist = left_shoulder + np.array([0.0, 0.47, 0.0])
        straight_elbow = left_shoulder + np.array([0.0, 0.25, 0.0])
        straight = SensorFrame(
            head=frame.head,
            waist=frame.waist,
            left_elbow=Pose(straight_elbow, arm_rotation(straight_elbow-left_shoulder)),
            right_elbow=frame.right_elbow,
            left_controller=Pose(straight_wrist, identity),
            right_controller=frame.right_controller,
            left_shoulder=frame.left_shoulder,
            right_shoulder=frame.right_shoulder,
        )
        straight_target = mapper.map(straight).left
        upper = straight_target.elbow.position-mapper.arms["left"].shoulder
        forearm = straight_target.wrist.position-straight_target.elbow.position
        self.assertAlmostEqual(
            float(np.dot(upper, forearm)/(np.linalg.norm(upper)*np.linalg.norm(forearm))),
            1.0,
            places=5,
        )
        raised_wrist = left_shoulder + np.array([0.0, 0.0, 0.47])
        raised = SensorFrame(
            head=frame.head,
            waist=frame.waist,
            left_elbow=frame.left_elbow,
            right_elbow=frame.right_elbow,
            left_controller=Pose(raised_wrist, identity),
            right_controller=frame.right_controller,
            left_shoulder=frame.left_shoulder,
            right_shoulder=frame.right_shoulder,
        )
        raised_target = mapper.map(raised).left
        raised_delta = raised_target.wrist.position - mapper.arms["left"].shoulder
        self.assertGreater(raised_delta[2], abs(raised_delta[1]))

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

    def test_one_packet_tracking_glitch_keeps_the_last_arm_pose(self):
        identity = Rotation.identity()
        left_shoulder = np.array([0.0, 0.15, 1.4])
        right_shoulder = np.array([0.0, -0.15, 1.4])
        left_elbow = left_shoulder + np.array([0.0, 0.25, 0.0])
        right_elbow = right_shoulder + np.array([0.0, -0.25, 0.0])
        left_wrist = left_elbow + np.array([0.22, 0.0, 0.0])
        right_wrist = right_elbow + np.array([0.22, 0.0, 0.0])

        def build(
            left_controller=left_wrist,
            right_controller=right_wrist,
            left_elbow_position=left_elbow,
            rotation=identity,
            shift=None,
        ):
            offset = np.zeros(3) if shift is None else shift

            def place(position, pose_rotation=identity):
                return Pose(rotation.apply(position) + offset, rotation * pose_rotation)

            return SensorFrame(
                head=place(0.5 * (left_shoulder + right_shoulder)),
                waist=place(np.zeros(3)),
                left_elbow=place(left_elbow_position),
                right_elbow=place(right_elbow),
                left_controller=place(left_controller),
                right_controller=place(right_controller),
                left_shoulder=place(left_shoulder),
                right_shoulder=place(right_shoulder),
            )

        base = build()
        mapper = PoseArmMapper(
            {
                "upper_arm_axis": "-y",
                "elbow_sensor_to_joint_m": 0.0,
                "head_position_filter_alpha": 1.0,
                "human_upper_arm_length_m": 0.30,
                "human_forearm_length_m": 0.26,
                "left_controller_to_wrist_m": [0.0, 0.0, 0.0],
                "right_controller_to_wrist_m": [0.0, 0.0, 0.0],
            },
            {"maximum_reach_ratio": 0.97, "reach_margin_m": 0.01},
            {"mode": "cooperative_wrist"},
        )
        robot = {
            "base": Pose(np.zeros(3), identity),
            "left_shoulder": Pose(np.array([0.0, 0.20, 1.40]), identity),
            "left_elbow": Pose(np.array([0.0, 0.60, 1.40]), identity),
            "left_wrist": Pose(np.array([0.35, 0.60, 1.40]), identity),
            "right_shoulder": Pose(np.array([0.0, -0.20, 1.40]), identity),
            "right_elbow": Pose(np.array([0.0, -0.60, 1.40]), identity),
            "right_wrist": Pose(np.array([0.35, -0.60, 1.40]), identity),
        }
        mapper.calibrate(base, robot)

        def plant(frame):
            mapper.accepted_sensor = frame
            mapper.previous_raw_sensor = frame
            mapper.sensor_stable_count = {"left": 4, "right": 4}
            mapper.sensor_slew_time = time.monotonic()

        plant(base)
        stepped = build(left_controller=left_wrist + np.array([0.04, 0.0, 0.0]))
        accepted = mapper.stabilize_sensor(stepped)
        self.assertIsNone(mapper.sensor_glitch)
        moved = mapper.map(accepted)
        self.assertFalse(mapper.at_calibration_pose)
        step_distance = np.linalg.norm(moved.left.wrist.position - robot["left_wrist"].position)
        self.assertGreater(step_distance, 0.01)
        self.assertLess(step_distance, 0.12)

        plant(base)
        jumped = build(left_controller=left_wrist + np.array([0.71, 0.0, 0.0]))
        held = mapper.stabilize_sensor(jumped)
        self.assertIn("left controller position", mapper.sensor_glitch)
        np.testing.assert_allclose(held.left_controller.position, base.left_controller.position)
        np.testing.assert_allclose(held.left_elbow.position, base.left_elbow.position)
        still_held = mapper.stabilize_sensor(jumped)
        np.testing.assert_allclose(still_held.left_controller.position, base.left_controller.position)
        mapped = mapper.map(held)
        np.testing.assert_allclose(mapped.left.wrist.position, robot["left_wrist"].position, atol=1e-4)

        twisted = build(
            left_controller=left_wrist + np.array([0.05, 0.0, 0.0]),
            left_elbow_position=left_elbow + np.array([0.0, 0.11, 0.0]),
        )
        twisted = SensorFrame(
            head=twisted.head,
            waist=twisted.waist,
            left_elbow=twisted.left_elbow,
            right_elbow=twisted.right_elbow,
            left_controller=Pose(
                twisted.left_controller.position,
                Rotation.from_euler("z", 60.0, degrees=True),
            ),
            right_controller=twisted.right_controller,
            left_shoulder=twisted.left_shoulder,
            right_shoulder=twisted.right_shoulder,
        )
        plant(base)
        filtered = mapper.stabilize_sensor(twisted)
        self.assertIn("left controller rotation", mapper.sensor_glitch)
        self.assertIn("left elbow", mapper.sensor_glitch)
        np.testing.assert_allclose(
            filtered.left_controller.position, left_wrist + np.array([0.05, 0.0, 0.0])
        )
        self.assertAlmostEqual(
            (filtered.left_controller.rotation.inv() * identity).magnitude(), 0.0, places=6
        )
        np.testing.assert_allclose(filtered.left_elbow.position, base.left_elbow.position)

        shifted = build(shift=np.array([0.40, -0.10, 0.05]))
        plant(base)
        followed = mapper.stabilize_sensor(shifted)
        self.assertIsNone(mapper.sensor_glitch)
        np.testing.assert_allclose(followed.left_controller.position, shifted.left_controller.position)

        yaw = Rotation.from_euler("z", 30.0, degrees=True)
        turned = build(rotation=yaw)
        plant(base)
        followed = mapper.stabilize_sensor(turned)
        self.assertIsNone(mapper.sensor_glitch)
        np.testing.assert_allclose(followed.left_controller.position, turned.left_controller.position)

        plant(base)
        target_rotation = Rotation.from_euler("z", 160.0, degrees=True)
        persistent = build(left_controller=left_wrist + np.array([0.71, 0.0, 0.0]))
        persistent = SensorFrame(
            head=persistent.head,
            waist=persistent.waist,
            left_elbow=persistent.left_elbow,
            right_elbow=persistent.right_elbow,
            left_controller=Pose(persistent.left_controller.position, target_rotation),
            right_controller=persistent.right_controller,
            left_shoulder=persistent.left_shoulder,
            right_shoulder=persistent.right_shoulder,
        )
        caught = mapper.stabilize_sensor(persistent)
        for _ in range(3):
            caught = mapper.stabilize_sensor(persistent)
        np.testing.assert_allclose(caught.left_controller.position, base.left_controller.position)
        mapper.sensor_slew_time = time.monotonic() - 0.028
        caught = mapper.stabilize_sensor(persistent)
        self.assertGreater(
            np.linalg.norm(caught.left_controller.position - base.left_controller.position),
            0.10,
        )
        self.assertLess(
            np.linalg.norm(caught.left_controller.position - persistent.left_controller.position),
            0.60,
        )
        caught_angle = np.degrees((caught.left_controller.rotation).magnitude())
        self.assertGreater(caught_angle, 30.0)
        self.assertLess(caught_angle, 50.0)
        for _ in range(8):
            mapper.sensor_slew_time = time.monotonic() - 0.028
            caught = mapper.stabilize_sensor(persistent)
        np.testing.assert_allclose(
            caught.left_controller.position,
            persistent.left_controller.position,
            atol=0.02,
        )
        self.assertLess(
            np.degrees((target_rotation.inv() * caught.left_controller.rotation).magnitude()),
            5.0,
        )
        self.assertIsNone(mapper.sensor_glitch)

    def test_a_shoulder_yaw_snap_does_not_swing_still_hands(self):
        identity = Rotation.identity()
        left_shoulder = np.array([0.0, 0.15, 1.4])
        right_shoulder = np.array([0.0, -0.15, 1.4])
        left_elbow = left_shoulder + np.array([0.0, 0.25, 0.0])
        right_elbow = right_shoulder + np.array([0.0, -0.25, 0.0])
        left_wrist = left_elbow + np.array([0.22, 0.0, 0.0])
        right_wrist = right_elbow + np.array([0.22, 0.0, 0.0])
        base = SensorFrame(
            head=Pose(0.5 * (left_shoulder + right_shoulder), identity),
            waist=Pose(np.zeros(3), identity),
            left_elbow=Pose(left_elbow, identity),
            right_elbow=Pose(right_elbow, identity),
            left_controller=Pose(left_wrist, identity),
            right_controller=Pose(right_wrist, identity),
            left_shoulder=Pose(left_shoulder, identity),
            right_shoulder=Pose(right_shoulder, identity),
        )
        mapper = PoseArmMapper(
            {
                "upper_arm_axis": "-y",
                "elbow_sensor_to_joint_m": 0.0,
                "head_position_filter_alpha": 1.0,
                "left_controller_to_wrist_m": [0.0, 0.0, 0.0],
                "right_controller_to_wrist_m": [0.0, 0.0, 0.0],
            },
            {"maximum_reach_ratio": 0.97, "reach_margin_m": 0.01},
            {"mode": "cooperative_wrist"},
        )
        robot = {
            "base": Pose(np.zeros(3), identity),
            "left_shoulder": Pose(np.array([0.0, 0.20, 1.40]), identity),
            "left_elbow": Pose(np.array([0.0, 0.60, 1.40]), identity),
            "left_wrist": Pose(np.array([0.35, 0.60, 1.40]), identity),
            "right_shoulder": Pose(np.array([0.0, -0.20, 1.40]), identity),
            "right_elbow": Pose(np.array([0.0, -0.60, 1.40]), identity),
            "right_wrist": Pose(np.array([0.35, -0.60, 1.40]), identity),
        }
        mapper.calibrate(base, robot)
        wrists = {
            side: mapper._controller_wrist_world(base, side) for side in ("left", "right")
        }
        before = mapper._map_cooperative(base, base.head.position, wrists)
        snapped = replace(
            base,
            waist=Pose(base.waist.position, Rotation.from_euler("z", 9.0, degrees=True)),
        )
        held = mapper.stabilize_sensor(snapped)
        self.assertIn("waist yaw", mapper.sensor_glitch)
        after = mapper._map_cooperative(held, held.head.position, wrists)
        np.testing.assert_allclose(after.left.wrist.position, before.left.wrist.position, atol=1e-6)
        np.testing.assert_allclose(after.right.wrist.position, before.right.wrist.position, atol=1e-6)
        still = mapper.stabilize_sensor(snapped)
        self.assertIn("waist yaw", mapper.sensor_glitch)
        self.assertAlmostEqual(mapper.yaw_of(still.waist.rotation), 0.0, places=6)

        turned = replace(
            base,
            waist=Pose(base.waist.position, Rotation.from_euler("z", 3.0, degrees=True)),
            left_controller=Pose(left_wrist + np.array([0.05, 0.0, 0.0]), identity),
            right_controller=Pose(right_wrist + np.array([0.05, 0.0, 0.0]), identity),
        )
        mapper.accepted_sensor = base
        mapper.previous_raw_sensor = base
        followed = mapper.stabilize_sensor(turned)
        self.assertIsNone(mapper.sensor_glitch)
        self.assertAlmostEqual(
            mapper.yaw_of(followed.waist.rotation), np.radians(3.0), places=5
        )


if __name__ == "__main__":
    unittest.main()

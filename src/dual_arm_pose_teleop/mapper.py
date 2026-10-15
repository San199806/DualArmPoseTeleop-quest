from dataclasses import dataclass, replace
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import axis_vector, make_arm_frame, normalize, project_two_link
from .types import ArmTarget, DualArmTargets, Pose, SensorFrame


@dataclass
class ArmCalibration:
    direction_alignment: Rotation
    controller_to_wrist: Rotation
    head_to_shoulder_body: np.ndarray
    shoulder: np.ndarray
    upper_length: float
    forearm_length: float
    last_elbow_direction: np.ndarray
    wrist_position_offset: np.ndarray


class PoseArmMapper:
    """Map head-anchored human arm directions to robot link targets."""

    def __init__(self, sensor_config, ik_config, mapping_config=None):
        mapping_config = mapping_config or {}
        self.mode = mapping_config.get("mode", "segment_direction")
        self.last_feasible_wrists = None
        self.last_targets = None
        if self.mode not in ("segment_direction", "cooperative_wrist"):
            raise ValueError(f"Unsupported mapping mode: {self.mode}")
        self.upper_axis = axis_vector(sensor_config["upper_arm_axis"])
        self.elbow_offset = float(sensor_config["elbow_sensor_to_joint_m"])
        default_human_upper = float(
            sensor_config.get("human_upper_arm_length_m", 0.30)
        )
        self.human_upper_lengths = {
            "left": float(
                sensor_config.get("left_human_upper_arm_length_m", default_human_upper)
            ),
            "right": float(
                sensor_config.get("right_human_upper_arm_length_m", default_human_upper)
            ),
        }
        default_human_forearm = float(
            sensor_config.get("human_forearm_length_m", 0.26)
        )
        self.human_forearm_lengths = {
            "left": float(
                sensor_config.get(
                    "left_human_forearm_length_m", default_human_forearm
                )
            ),
            "right": float(
                sensor_config.get(
                    "right_human_forearm_length_m", default_human_forearm
                )
            ),
        }
        self.human_shoulder_width = float(
            sensor_config.get("human_shoulder_width_m", 0.38)
        )
        if self.human_shoulder_width <= 0.0:
            raise ValueError("human_shoulder_width_m must be positive")
        self.head_filter_alpha = float(
            sensor_config.get("head_position_filter_alpha", 0.20)
        )
        if not 0.0 < self.head_filter_alpha <= 1.0:
            raise ValueError("head_position_filter_alpha must be in (0, 1]")
        self.controller_offsets = {
            "left": np.asarray(sensor_config["left_controller_to_wrist_m"], dtype=float),
            "right": np.asarray(sensor_config["right_controller_to_wrist_m"], dtype=float),
        }
        self.maximum_reach_ratio = float(ik_config["maximum_reach_ratio"])
        self.reach_margin = float(ik_config["reach_margin_m"])
        self.base_pose = None
        self.arms = {}
        self.filtered_head_position = None
        self.filtered_wrist_positions = {}
        self.body_alignment = None
        self.common_scale = None
        self.skeleton_segments = False
        self.calibration_wrist_vectors = {}
        self.calibration_targets = None
        self.calibration_yaw = None
        self.calibration_pitch = None
        self.at_calibration_pose = False
        self.last_debug_metrics = {}
        self.accepted_sensor = None
        self.previous_raw_sensor = None
        self.sensor_stable_count = {"left": 0, "right": 0}
        self.sensor_slew_time = 0.0
        self.sensor_glitch = None

    @property
    def calibrated(self):
        return self.base_pose is not None and len(self.arms) == 2

    def reset(self):
        self.last_feasible_wrists = None
        self.last_targets = None
        self.base_pose = None
        self.arms.clear()
        self.filtered_head_position = None
        self.filtered_wrist_positions.clear()
        self.body_alignment = None
        self.common_scale = None
        self.skeleton_segments = False
        self.calibration_wrist_vectors = {}
        self.calibration_targets = None
        self.calibration_yaw = None
        self.calibration_pitch = None
        self.at_calibration_pose = False
        self.last_debug_metrics = {}
        self.accepted_sensor = None
        self.previous_raw_sensor = None
        self.sensor_stable_count = {"left": 0, "right": 0}
        self.sensor_slew_time = 0.0
        self.sensor_glitch = None

    def _adopt_skeleton_segments(self, frame):
        """Use the VR shoulder, elbow and wrist bones as the human arm lengths."""
        self.skeleton_segments = False
        shoulders = {}
        for side in ("left", "right"):
            shoulder = getattr(frame, f"{side}_shoulder", None)
            if shoulder is None:
                return
            shoulders[side] = shoulder.position
            elbow = getattr(frame, f"{side}_elbow").position
            wrist = self._controller_wrist_world(frame, side)
            upper = float(np.linalg.norm(elbow-shoulders[side]))
            forearm = float(np.linalg.norm(wrist-elbow))
            if upper < 0.05 or forearm < 0.05:
                return
            self.human_upper_lengths[side] = upper
            self.human_forearm_lengths[side] = forearm
        lateral = shoulders["left"]-shoulders["right"]
        lateral[2] = 0.0
        width = float(np.linalg.norm(lateral))
        if width > 0.05:
            self.human_shoulder_width = width
        # A straightened human arm maps onto the robot's full link length.
        # The default 97% reach cap would reject that pose and keep the last bend.
        self.maximum_reach_ratio = 1.0
        self.reach_margin = 0.0
        self.skeleton_segments = True

    @staticmethod
    def _mean_pose(poses):
        positions = np.stack([pose.position for pose in poses])
        rotations = Rotation.from_quat(
            np.stack([pose.rotation.as_quat() for pose in poses])
        )
        return Pose(np.mean(positions, axis=0), rotations.mean())

    @classmethod
    def average_frame(cls, frames):
        """Average a short, stationary calibration window."""
        if not frames:
            raise ValueError("Cannot average an empty calibration window")
        def mean_optional(name):
            poses = [getattr(frame, name) for frame in frames]
            if any(pose is None for pose in poses):
                return None
            return cls._mean_pose(poses)

        return SensorFrame(
            head=cls._mean_pose([frame.head for frame in frames]),
            waist=cls._mean_pose([frame.waist for frame in frames]),
            left_elbow=cls._mean_pose([frame.left_elbow for frame in frames]),
            right_elbow=cls._mean_pose([frame.right_elbow for frame in frames]),
            left_controller=cls._mean_pose([frame.left_controller for frame in frames]),
            right_controller=cls._mean_pose([frame.right_controller for frame in frames]),
            left_shoulder=mean_optional("left_shoulder"),
            right_shoulder=mean_optional("right_shoulder"),
            hips=mean_optional("hips"),
            chest=mean_optional("chest"),
        )

    def calibration_motion_deg(self, frames):
        """Return the largest waist-relative arm change in a sample window."""
        if len(frames) < 2:
            return 0.0
        reference = {}
        for side in ("left", "right"):
            upper, controller = self._orientation_features(frames[0], side)
            reference[side] = (upper, controller)

        maximum = 0.0
        for frame in frames[1:]:
            for side in ("left", "right"):
                upper, controller = self._orientation_features(frame, side)
                upper0, controller0 = reference[side]
                upper_angle = np.degrees(
                    np.arccos(np.clip(np.dot(upper0, upper), -1.0, 1.0))
                )
                controller_angle = np.degrees(
                    (controller0.inv() * controller).magnitude()
                )
                maximum = max(maximum, upper_angle, controller_angle)
        return float(maximum)

    @staticmethod
    def calibration_position_motion_m(frames):
        """Largest positional excursion during a calibration window."""
        if len(frames) < 2:
            return 0.0
        maximum = 0.0
        for name in ("head", "left_controller", "right_controller"):
            positions = np.stack([getattr(frame, name).position for frame in frames])
            center = np.mean(positions, axis=0)
            maximum = max(
                maximum,
                float(np.max(np.linalg.norm(positions - center, axis=1))),
            )
        return maximum

    def calibration_geometry(self, frame):
        left = self._controller_wrist_world(frame, "left")
        right = self._controller_wrist_world(frame, "right")
        left_horizontal = (left - frame.head.position).copy()
        right_horizontal = (right - frame.head.position).copy()
        left_horizontal[2] = 0.0
        right_horizontal[2] = 0.0
        return {
            "hand_separation_m": float(np.linalg.norm(left - right)),
            "hand_height_diff_m": float(abs(left[2] - right[2])),
            "hand_radial_diff_m": float(
                abs(np.linalg.norm(left_horizontal) - np.linalg.norm(right_horizontal))
            ),
        }

    def _orientation_features(self, frame, side):
        elbow_sensor = getattr(frame, f"{side}_elbow")
        controller = getattr(frame, f"{side}_controller")
        waist_inv = frame.waist.rotation.inv()
        sensor_rotation_local = waist_inv * elbow_sensor.rotation
        toward_shoulder = sensor_rotation_local.apply(self.upper_axis)
        upper_direction = normalize(-toward_shoulder)
        controller_rotation_local = waist_inv * controller.rotation
        return upper_direction, controller_rotation_local

    def _controller_wrist_world(self, frame, side):
        controller = getattr(frame, f"{side}_controller")
        return controller.position + controller.rotation.apply(
            self.controller_offsets[side]
        )

    def _calibration_observation(self, frame, side):
        """Estimate one raw shoulder before bilateral symmetrization."""
        if self.skeleton_segments and getattr(frame, f"{side}_shoulder", None) is not None:
            upper, _, controller_rotation, shoulder_world, _ = self._human_arm(
                frame, side, frame.head.position
            )
            return upper, controller_rotation, shoulder_world
        elbow_sensor = getattr(frame, f"{side}_elbow")
        upper_direction, controller_rotation_local = self._orientation_features(
            frame, side
        )
        upper_world = frame.waist.rotation.apply(upper_direction)
        elbow_world = elbow_sensor.position + upper_world * self.elbow_offset
        raw_shoulder_world = (
            elbow_world - self.human_upper_lengths[side] * upper_world
        )
        return upper_direction, controller_rotation_local, raw_shoulder_world

    def _symmetric_shoulders(self, frame, observations):
        """Create equal-height shoulders using the accurately tracked controllers."""
        raw_left = observations["left"][2]
        raw_right = observations["right"][2]
        center = 0.5 * (raw_left + raw_right)

        left_wrist = self._controller_wrist_world(frame, "left")
        right_wrist = self._controller_wrist_world(frame, "right")
        lateral = left_wrist - right_wrist
        lateral[2] = 0.0
        fallback = raw_left - raw_right
        fallback[2] = 0.0
        lateral = normalize(
            lateral,
            fallback if np.linalg.norm(fallback) > 1e-8 else [1.0, 0.0, 0.0],
        )
        half_width = 0.5 * self.human_shoulder_width
        return {
            "left": center + half_width * lateral,
            "right": center - half_width * lateral,
        }

    @staticmethod
    def _body_frame(lateral_direction, up_direction):
        """Build a right-handed body frame from semantic left and up axes."""
        lateral = normalize(lateral_direction)
        up = np.asarray(up_direction, dtype=np.float64).copy()
        up -= np.dot(up, lateral) * lateral
        up = normalize(up, [0.0, 0.0, 1.0])
        forward = normalize(np.cross(up, lateral))
        up = normalize(np.cross(lateral, forward))
        return Rotation.from_matrix(
            np.column_stack([lateral, forward, up])
        )

    def _human_arm(self, frame, side, head_position, wrist_world=None):
        """Reconstruct the elbow without using its runtime tracked position.

        A VR skeleton supplies the shoulder bone, so its upper-arm and forearm
        directions come from the measured segments instead.
        """
        if self.skeleton_segments and getattr(frame, f"{side}_shoulder", None) is not None:
            elbow_pose = getattr(frame, f"{side}_elbow")
            controller = getattr(frame, f"{side}_controller")
            if wrist_world is None:
                wrist_world = self._controller_wrist_world(frame, side)
            shoulder_world = getattr(frame, f"{side}_shoulder").position
            waist_inv = frame.waist.rotation.inv()
            upper_direction = normalize(waist_inv.apply(elbow_pose.position-shoulder_world))
            forearm_direction = normalize(
                waist_inv.apply(wrist_world-elbow_pose.position), upper_direction
            )
            return (
                upper_direction,
                forearm_direction,
                waist_inv * controller.rotation,
                shoulder_world,
                wrist_world,
            )
        calibration = self.arms[side]
        upper_direction, controller_rotation_local = self._orientation_features(
            frame, side
        )
        upper_world = frame.waist.rotation.apply(upper_direction)
        shoulder_world = head_position + frame.waist.rotation.apply(
            calibration.head_to_shoulder_body
        )
        elbow_world = (
            shoulder_world + self.human_upper_lengths[side] * upper_world
        )
        if wrist_world is None:
            wrist_world = self._controller_wrist_world(frame, side)
        forearm_direction = normalize(
            frame.waist.rotation.inv().apply(wrist_world - elbow_world),
            upper_direction,
        )
        return (
            upper_direction,
            forearm_direction,
            controller_rotation_local,
            shoulder_world,
            wrist_world,
        )

    def _filter_head_position(self, position):
        position = np.asarray(position, dtype=np.float64)
        if self.filtered_head_position is None:
            self.filtered_head_position = position.copy()
        else:
            alpha = self.head_filter_alpha
            self.filtered_head_position = (
                (1.0 - alpha) * self.filtered_head_position + alpha * position
            )
        return self.filtered_head_position.copy()

    def _filter_wrist_position(self, side, position):
        position = np.asarray(position, dtype=np.float64)
        if side not in self.filtered_wrist_positions:
            self.filtered_wrist_positions[side] = position.copy()
        else:
            alpha = self.head_filter_alpha
            self.filtered_wrist_positions[side] = (
                (1.0 - alpha) * self.filtered_wrist_positions[side]
                + alpha * position
            )
        return self.filtered_wrist_positions[side].copy()

    @staticmethod
    def _to_local(pose, base_pose):
        return Pose(
            base_pose.rotation.inv().apply(pose.position - base_pose.position),
            base_pose.rotation.inv() * pose.rotation,
        )

    def calibrate(self, frame, robot_poses):
        self._adopt_skeleton_segments(frame)
        self.base_pose = robot_poses["base"]
        self.filtered_head_position = frame.head.position.copy()
        self.filtered_wrist_positions = {
            side: self._controller_wrist_world(frame, side).copy()
            for side in ("left", "right")
        }
        observations = {
            side: self._calibration_observation(frame, side)
            for side in ("left", "right")
        }
        human_shoulders = self._symmetric_shoulders(frame, observations)
        waist_inv = frame.waist.rotation.inv()
        robot_shoulders = {
            side: self._to_local(
                robot_poses[f"{side}_shoulder"], self.base_pose
            ).position
            for side in ("left", "right")
        }
        human_lateral_local = waist_inv.apply(
            human_shoulders["left"] - human_shoulders["right"]
        )
        human_up_local = waist_inv.apply(np.array([0.0, 0.0, 1.0]))
        robot_lateral_local = (
            robot_shoulders["left"] - robot_shoulders["right"]
        )
        robot_up_local = self.base_pose.rotation.inv().apply(
            np.array([0.0, 0.0, 1.0])
        )
        human_body_frame = self._body_frame(
            human_lateral_local, human_up_local
        )
        robot_body_frame = self._body_frame(
            robot_lateral_local, robot_up_local
        )
        self.body_alignment = robot_body_frame * human_body_frame.inv()
        robot_arm_lengths = []
        human_arm_lengths = []
        for side in ("left", "right"):
            shoulder = self._to_local(robot_poses[f"{side}_shoulder"], self.base_pose)
            elbow = self._to_local(robot_poses[f"{side}_elbow"], self.base_pose)
            wrist = self._to_local(robot_poses[f"{side}_wrist"], self.base_pose)
            robot_arm_lengths.append(
                np.linalg.norm(elbow.position - shoulder.position)
                + np.linalg.norm(wrist.position - elbow.position)
            )
            human_arm_lengths.append(
                self.human_upper_lengths[side] + self.human_forearm_lengths[side]
            )
        self.common_scale = float(np.mean(robot_arm_lengths) / np.mean(human_arm_lengths))
        for side in ("left", "right"):
            shoulder = self._to_local(robot_poses[f"{side}_shoulder"], self.base_pose)
            elbow = self._to_local(robot_poses[f"{side}_elbow"], self.base_pose)
            wrist = self._to_local(robot_poses[f"{side}_wrist"], self.base_pose)
            human_upper, controller_rotation, _ = observations[side]
            human_shoulder_world = human_shoulders[side]
            human_upper_world = frame.waist.rotation.apply(human_upper)
            reconstructed_elbow_world = (
                human_shoulder_world
                + self.human_upper_lengths[side] * human_upper_world
            )
            human_wrist_world = self._controller_wrist_world(frame, side)
            vector_shoulder = human_shoulder_world
            vector_wrist = human_wrist_world
            # Runtime mapping reads the skeleton shoulder. The offset has to use
            # that same point, or the calibrated L pose maps tens of centimetres away.
            if self.skeleton_segments and getattr(frame, f"{side}_shoulder", None) is not None:
                _, _, _, vector_shoulder, vector_wrist = self._human_arm(
                    frame, side, frame.head.position
                )
            if self.skeleton_segments:
                human_forearm = self._human_arm(frame, side, frame.head.position)[1]
            else:
                human_forearm = normalize(
                    waist_inv.apply(human_wrist_world - reconstructed_elbow_world),
                    human_upper,
                )
            head_to_shoulder_body = waist_inv.apply(
                human_shoulder_world - frame.head.position
            )
            robot_upper = elbow.position - shoulder.position
            robot_forearm = wrist.position - elbow.position
            human_frame = make_arm_frame(human_upper, human_forearm)
            robot_frame = make_arm_frame(robot_upper, robot_forearm)
            # Arm attitude calibration remains side-specific so the session-zero
            # elbow and wrist attitude reproduce the robot's initial pose.  Only
            # cooperative wrist position mapping uses the common body alignment.
            alignment = robot_frame * human_frame.inv()
            attitude_alignment = self.body_alignment if self.mode == "cooperative_wrist" else alignment
            controller_in_robot = attitude_alignment * controller_rotation
            controller_to_wrist = controller_in_robot.inv() * wrist.rotation
            if self.skeleton_segments and self.mode == "segment_direction":
                predicted_wrist = (
                    shoulder.position
                    + float(np.linalg.norm(robot_upper)) * normalize(alignment.apply(human_upper))
                    + float(np.linalg.norm(robot_forearm)) * normalize(alignment.apply(human_forearm))
                )
            else:
                human_wrist_vector_local = waist_inv.apply(
                    vector_wrist - vector_shoulder
                )
                predicted_wrist = shoulder.position + self.common_scale * self.body_alignment.apply(
                    human_wrist_vector_local
                )
            wrist_position_offset = wrist.position - predicted_wrist
            elbow_direction = elbow.position - 0.5 * (shoulder.position + wrist.position)
            self.arms[side] = ArmCalibration(
                direction_alignment=alignment,
                controller_to_wrist=controller_to_wrist,
                head_to_shoulder_body=head_to_shoulder_body,
                shoulder=shoulder.position,
                upper_length=float(np.linalg.norm(robot_upper)),
                forearm_length=float(np.linalg.norm(robot_forearm)),
                last_elbow_direction=normalize(elbow_direction, [0.0, 0.0, 1.0]),
                wrist_position_offset=wrist_position_offset,
            )
        self.last_feasible_wrists = {
            side: self._to_local(robot_poses[f"{side}_wrist"], self.base_pose).position.copy()
            for side in ("left", "right")
        }
        if self.mode == "cooperative_wrist" and not self._wrists_feasible(self.last_feasible_wrists):
            self.reset()
            raise ValueError("initial_qpos wrist targets exceed configured reach margins; bend the arms before calibrating")
        self.last_targets = DualArmTargets(**{
            side: ArmTarget(robot_poses[f"{side}_elbow"], robot_poses[f"{side}_wrist"])
            for side in ("left", "right")
        })
        self.calibration_targets = self.last_targets
        self.calibration_wrist_vectors = {}
        for side in ("left", "right"):
            _, _, _, shoulder, wrist = self._human_arm(frame, side, frame.head.position)
            self.calibration_wrist_vectors[side] = frame.waist.rotation.inv().apply(
                wrist - shoulder
            )
        self.at_calibration_pose = True
        self.accepted_sensor = frame
        self.previous_raw_sensor = frame
        self.sensor_stable_count = {"left": 4, "right": 4}
        self.sensor_slew_time = time.monotonic()
        self.sensor_glitch = None
        self.calibration_yaw = self.yaw_of(frame.waist.rotation)
        self.calibration_pitch = self._frame_torso_pitch(frame)

    @staticmethod
    def yaw_of(rotation):
        forward = rotation.apply(np.array([1.0, 0.0, 0.0]))
        return float(np.arctan2(forward[1], forward[0]))

    @staticmethod
    def torso_pitch(hips, chest, yaw):
        """Positive when the chest leans forward of the hips, yaw removed."""
        delta = np.asarray(chest, dtype=float) - np.asarray(hips, dtype=float)
        local = Rotation.from_euler("z", -yaw).apply(delta)
        return float(np.arctan2(local[0], local[2]))

    @classmethod
    def _frame_torso_pitch(cls, frame):
        if frame.hips is None or frame.chest is None:
            return None
        return cls.torso_pitch(
            frame.hips.position, frame.chest.position, cls.yaw_of(frame.waist.rotation)
        )

    @staticmethod
    def torso_joint_target(current_yaw, calibration_yaw, initial, lower, upper, current_joint, max_step):
        """Waist joint for the change in body yaw since calibration, rate-limited."""
        delta = (current_yaw - calibration_yaw + np.pi) % (2.0 * np.pi) - np.pi
        desired = float(np.clip(initial + delta, lower, upper))
        return float(np.clip(desired, current_joint - max_step, current_joint + max_step))

    def export_calibration(self):
        if not self.calibrated:
            raise RuntimeError("Cannot export an incomplete calibration")
        return {
            "version": 5,
            "mode": self.mode,
            "body_alignment_xyzw": self.body_alignment.as_quat().tolist(),
            "common_scale": self.common_scale,
            "arms": {
                side: {
                    "direction_alignment_xyzw": self.arms[side]
                    .direction_alignment.as_quat()
                    .tolist(),
                    "controller_to_wrist_xyzw": self.arms[side]
                    .controller_to_wrist.as_quat()
                    .tolist(),
                    "head_to_shoulder_body": self.arms[side]
                    .head_to_shoulder_body.tolist(),
                    "wrist_position_offset": self.arms[side]
                    .wrist_position_offset.tolist(),
                }
                for side in ("left", "right")
            },
        }

    def load_calibration(self, saved, robot_poses):
        """Load stable sensor rotations while refreshing geometry from robot FK."""
        if saved.get("version") != 5:
            raise ValueError("Unsupported calibration version")
        if saved.get("mode") != self.mode:
            raise ValueError("Calibration mapping mode does not match")
        self.body_alignment = Rotation.from_quat(
            saved["body_alignment_xyzw"]
        )
        self.common_scale = float(saved["common_scale"])
        saved_arms = saved.get("arms", {})
        self.base_pose = robot_poses["base"]
        loaded = {}
        for side in ("left", "right"):
            values = saved_arms.get(side, {})
            shoulder = self._to_local(
                robot_poses[f"{side}_shoulder"], self.base_pose
            )
            elbow = self._to_local(robot_poses[f"{side}_elbow"], self.base_pose)
            wrist = self._to_local(robot_poses[f"{side}_wrist"], self.base_pose)
            robot_upper = elbow.position - shoulder.position
            robot_forearm = wrist.position - elbow.position
            elbow_direction = elbow.position - 0.5 * (
                shoulder.position + wrist.position
            )
            loaded[side] = ArmCalibration(
                direction_alignment=Rotation.from_quat(
                    values["direction_alignment_xyzw"]
                ),
                controller_to_wrist=Rotation.from_quat(
                    values["controller_to_wrist_xyzw"]
                ),
                head_to_shoulder_body=np.asarray(
                    values["head_to_shoulder_body"], dtype=np.float64
                ),
                shoulder=shoulder.position,
                upper_length=float(np.linalg.norm(robot_upper)),
                forearm_length=float(np.linalg.norm(robot_forearm)),
                last_elbow_direction=normalize(
                    elbow_direction, [0.0, 0.0, 1.0]
                ),
                wrist_position_offset=np.asarray(
                    values["wrist_position_offset"], dtype=np.float64
                ),
            )
        self.arms = loaded
        self.filtered_head_position = None
        self.filtered_wrist_positions.clear()
        self.last_feasible_wrists = {
            side: self._to_local(robot_poses[f"{side}_wrist"], self.base_pose).position.copy()
            for side in ("left", "right")
        }
        if self.mode == "cooperative_wrist" and not self._wrists_feasible(self.last_feasible_wrists):
            self.reset()
            raise ValueError("Calibration reference exceeds reach margins")
        self.last_targets = DualArmTargets(**{
            side: ArmTarget(robot_poses[f"{side}_elbow"], robot_poses[f"{side}_wrist"])
            for side in ("left", "right")
        })

    def _wrists_feasible(self, wrists):
        for side, wrist in wrists.items():
            arm = self.arms[side]
            distance = np.linalg.norm(wrist-arm.shoulder)
            span = arm.upper_length+arm.forearm_length
            if self.skeleton_segments:
                minimum = abs(arm.upper_length-arm.forearm_length)
                # A straightened arm can sit a few millimetres past the robot's
                # full length. Clipping later keeps that direction; rejecting
                # here would freeze the previous pose.
                maximum = span*2.0
            else:
                minimum = abs(arm.upper_length-arm.forearm_length)+self.reach_margin
                maximum = min(span*self.maximum_reach_ratio, span-self.reach_margin)
            if not np.isfinite(distance) or not minimum-1e-9 <= distance <= maximum+1e-9:
                return False
        return True

    def _map_arm(self, frame, side, head_position, wrist_world=None):
        calibration = self.arms[side]
        (
            upper_h,
            forearm_h,
            controller_rotation,
            human_shoulder_world,
            human_wrist_world,
        ) = self._human_arm(frame, side, head_position, wrist_world)
        if self.mode == "cooperative_wrist":
            upper_r = normalize(calibration.direction_alignment.apply(upper_h))
            raw_elbow = (
                calibration.shoulder + calibration.upper_length * upper_r
            )
            human_wrist_vector_local = frame.waist.rotation.inv().apply(
                human_wrist_world - human_shoulder_world
            )
            raw_wrist = (
                calibration.shoulder
                + self.common_scale * self.body_alignment.apply(human_wrist_vector_local)
                + calibration.wrist_position_offset
            )
        else:
            upper_r = normalize(calibration.direction_alignment.apply(upper_h))
            forearm_r = normalize(
                calibration.direction_alignment.apply(forearm_h)
            )
            raw_elbow = (
                calibration.shoulder + calibration.upper_length * upper_r
            )
            raw_wrist = raw_elbow + calibration.forearm_length * forearm_r
        elbow, wrist, elbow_direction = project_two_link(
            calibration.shoulder,
            raw_elbow,
            raw_wrist,
            calibration.upper_length,
            calibration.forearm_length,
            self.maximum_reach_ratio,
            self.reach_margin,
            calibration.last_elbow_direction,
        )
        calibration.last_elbow_direction = elbow_direction

        controller_in_robot = calibration.direction_alignment * controller_rotation
        wrist_rotation = controller_in_robot * calibration.controller_to_wrist
        elbow_world = self.base_pose.position + self.base_pose.rotation.apply(elbow)
        wrist_world = self.base_pose.position + self.base_pose.rotation.apply(wrist)
        wrist_rotation_world = self.base_pose.rotation * wrist_rotation
        return ArmTarget(
            elbow=Pose(elbow_world, Rotation.identity()),
            wrist=Pose(wrist_world, wrist_rotation_world),
        )

    def _couple_raw_wrists(self, raw_wrists):
        """Hold the previous pair if either wrist exceeds its reach shell."""
        # Strict collaboration: a rigid grasp must never be resized to fit.
        # Check both inner and outer reach bounds; keep the previous pair
        # whenever either arm is infeasible. There is no iterative convergence
        # heuristic and no independent left/right clipping here.
        if self._wrists_feasible(raw_wrists):
            self.last_feasible_wrists = {s: p.copy() for s, p in raw_wrists.items()}
        return (
            {s: p.copy() for s, p in self.last_feasible_wrists.items()},
            1.0,
            float(np.linalg.norm(raw_wrists["left"]-raw_wrists["right"])),
        )

    def _map_cooperative(self, frame, head_position, wrist_positions):
        observations = {}
        raw_wrists = {}
        raw_elbows = {}
        for side in ("left", "right"):
            values = self._human_arm(
                frame, side, head_position, wrist_positions[side]
            )
            observations[side] = values
            upper_h, _, _, human_shoulder_world, human_wrist_world = values
            arm = self.arms[side]
            upper_r = normalize(arm.direction_alignment.apply(upper_h))
            raw_elbows[side] = arm.shoulder + arm.upper_length * upper_r
            human_vector = frame.waist.rotation.inv().apply(
                human_wrist_world - human_shoulder_world
            )
            raw_wrists[side] = (
                arm.shoulder
                + self.common_scale * self.body_alignment.apply(human_vector)
                + arm.wrist_position_offset
            )

        if not self._wrists_feasible(raw_wrists):
            self.last_debug_metrics = {
                "mapping_workspace_clamped": 1.0,
                "mapping_pair_held": 1.0,
                "mapping_desired_wrist_distance_m": float(np.linalg.norm(raw_wrists["left"]-raw_wrists["right"])),
                "mapping_projected_wrist_distance_m": float(np.linalg.norm(self.last_targets.left.wrist.position-self.last_targets.right.wrist.position)),
            }
            return self.last_targets
        coupled_wrists, relative_scale, desired_distance = self._couple_raw_wrists(
            raw_wrists
        )
        targets = {}
        projected_wrists = {}
        for side in ("left", "right"):
            arm = self.arms[side]
            reach_ratio = 1.0 if self.skeleton_segments else self.maximum_reach_ratio
            reach_margin = 0.0 if self.skeleton_segments else self.reach_margin
            elbow, wrist, elbow_direction = project_two_link(
                arm.shoulder,
                raw_elbows[side],
                coupled_wrists[side],
                arm.upper_length,
                arm.forearm_length,
                reach_ratio,
                reach_margin,
                arm.last_elbow_direction,
            )
            arm.last_elbow_direction = elbow_direction
            controller_rotation = observations[side][2]
            wrist_rotation = (
                self.body_alignment * controller_rotation
                * arm.controller_to_wrist
            )
            targets[side] = ArmTarget(
                elbow=Pose(
                    self.base_pose.position + self.base_pose.rotation.apply(elbow),
                    Rotation.identity(),
                ),
                wrist=Pose(
                    self.base_pose.position + self.base_pose.rotation.apply(wrist),
                    self.base_pose.rotation * wrist_rotation,
                ),
            )
            projected_wrists[side] = wrist

        actual_distance = float(
            np.linalg.norm(projected_wrists["left"] - projected_wrists["right"])
        )
        self.last_debug_metrics = {
            "mapping_pair_held": 0.0,
            "mapping_common_scale": self.common_scale,
            "mapping_relative_scale": relative_scale,
            "mapping_desired_wrist_distance_m": desired_distance,
            "mapping_projected_wrist_distance_m": actual_distance,
            "mapping_relative_distance_error_m": abs(actual_distance - desired_distance),
            "mapping_workspace_clamped": float(
                relative_scale < 0.999
                or any(
                    np.linalg.norm(projected_wrists[side] - raw_wrists[side]) > 1e-4
                    for side in ("left", "right")
                )
            ),
        }
        for side in ("left", "right"):
            arm = self.arms[side]
            arm_length = arm.upper_length + arm.forearm_length
            self.last_debug_metrics.update(
                {
                    f"mapping_{side}_raw_reach_ratio": float(
                        np.linalg.norm(raw_wrists[side] - arm.shoulder) / arm_length
                    ),
                    f"mapping_{side}_projected_reach_ratio": float(
                        np.linalg.norm(projected_wrists[side] - arm.shoulder)
                        / arm_length
                    ),
                    f"mapping_{side}_projection_error_m": float(
                        np.linalg.norm(projected_wrists[side] - raw_wrists[side])
                    ),
                }
            )
        self.last_targets = DualArmTargets(left=targets["left"], right=targets["right"])
        return self.last_targets

    def _mark_calibration_pose(self, frame, wrist_positions):
        """True when both hands are back at the calibrated L pose."""
        if len(self.calibration_wrist_vectors) != 2:
            self.at_calibration_pose = False
            return
        for side in ("left", "right"):
            _, _, _, shoulder, wrist = self._human_arm(
                frame, side, frame.head.position, wrist_positions[side]
            )
            vector = frame.waist.rotation.inv().apply(wrist - shoulder)
            if np.linalg.norm(vector - self.calibration_wrist_vectors[side]) > 0.01:
                self.at_calibration_pose = False
                return
        self.at_calibration_pose = True

    def _sensor_anchor(self, frame, side):
        shoulder = getattr(frame, f"{side}_shoulder", None)
        if shoulder is not None:
            return np.asarray(shoulder.position, dtype=np.float64)
        return np.asarray(frame.head.position, dtype=np.float64)

    def _local_point(self, frame, side, position):
        return frame.waist.rotation.inv().apply(
            np.asarray(position, dtype=np.float64) - self._sensor_anchor(frame, side)
        )

    def _local_rotation(self, frame, rotation):
        return frame.waist.rotation.inv() * rotation

    def _reexpress_point(self, source, dest, side, position):
        return self._sensor_anchor(dest, side) + dest.waist.rotation.apply(
            self._local_point(source, side, position)
        )

    def _reexpress_rotation(self, source, dest, rotation):
        return dest.waist.rotation * self._local_rotation(source, rotation)

    def _arm_deltas(self, current, previous, side):
        current_controller = getattr(current, f"{side}_controller")
        previous_controller = getattr(previous, f"{side}_controller")
        position_delta = float(np.linalg.norm(
            self._local_point(current, side, current_controller.position)
            - self._local_point(previous, side, previous_controller.position)
        ))
        rotation_delta = float(np.degrees((
            self._local_rotation(previous, previous_controller.rotation).inv()
            * self._local_rotation(current, current_controller.rotation)
        ).magnitude()))
        elbow_delta = float(np.linalg.norm(
            self._local_point(current, side, getattr(current, f"{side}_elbow").position)
            - self._local_point(previous, side, getattr(previous, f"{side}_elbow").position)
        ))
        return position_delta, rotation_delta, elbow_delta

    @staticmethod
    def _slew_point(accepted, target, budget):
        delta = target - accepted
        distance = float(np.linalg.norm(delta))
        if distance <= budget or distance < 1e-9:
            return target, False
        return accepted + delta * (budget / distance), True

    @staticmethod
    def _slew_rotation(accepted, target, budget_deg):
        delta = accepted.inv() * target
        angle = float(np.degrees(delta.magnitude()))
        if angle <= budget_deg or angle < 1e-6:
            return target, False
        rotvec = delta.as_rotvec()
        rotvec = rotvec * (np.radians(budget_deg) / np.linalg.norm(rotvec))
        return accepted * Rotation.from_rotvec(rotvec), True

    def _hold_false_waist_yaw(self, frame, accepted):
        """Keep the last shoulder yaw when it snaps but the hands stay put.

        A one-packet shoulder-line correction rotates both arm targets even
        though the controllers did not move. A real turn moves the hands.
        """
        yaw_delta = self.yaw_of(frame.waist.rotation) - self.yaw_of(accepted.waist.rotation)
        yaw_delta = (yaw_delta + np.pi) % (2.0 * np.pi) - np.pi
        yaw_deg = abs(float(np.degrees(yaw_delta)))
        if yaw_deg <= 5.0:
            return frame, None
        hand_motion = max(
            float(np.linalg.norm(
                frame.left_controller.position - accepted.left_controller.position
            )),
            float(np.linalg.norm(
                frame.right_controller.position - accepted.right_controller.position
            )),
        )
        if hand_motion >= 0.04:
            return frame, None
        held = replace(
            frame,
            waist=Pose(
                np.asarray(frame.waist.position, dtype=np.float64),
                accepted.waist.rotation,
            ),
        )
        return held, f"waist yaw {yaw_deg:.0f} deg"

    def stabilize_sensor(self, frame):
        """Drop a one-packet arm glitch, then catch up once the pose settles.

        Motion is measured relative to the shoulder in the waist frame, so a
        step of the whole body still passes. A controller that teleports, a
        wrist that twists far in one packet, or an elbow that jumps while the
        hand stays still is ignored. Once the new reading stays put, the
        accepted pose moves toward it at the same per-packet limit.
        """
        previous_raw = self.previous_raw_sensor
        accepted = self.accepted_sensor
        if previous_raw is None or accepted is None:
            self.accepted_sensor = frame
            self.previous_raw_sensor = frame
            self.sensor_stable_count = {"left": 4, "right": 4}
            self.sensor_slew_time = time.monotonic()
            self.sensor_glitch = None
            return frame
        now = time.monotonic()
        elapsed = now - self.sensor_slew_time
        if elapsed < 0.001:
            scale = 0.0
        else:
            scale = min(elapsed, 0.04) * 36.0
            self.sensor_slew_time = now
        position_budget = 0.15 * scale
        rotation_budget = 40.0 * scale
        elbow_budget = 0.08 * scale
        frame, yaw_reason = self._hold_false_waist_yaw(frame, accepted)
        replaces = {}
        reasons = []
        for side in ("left", "right"):
            position_delta, rotation_delta, elbow_delta = self._arm_deltas(
                frame, previous_raw, side
            )
            position_jump = position_delta > 0.15
            rotation_jump = rotation_delta > 40.0
            elbow_jump = elbow_delta > 0.08 and position_delta < 0.06
            if position_jump or rotation_jump or elbow_jump:
                self.sensor_stable_count[side] = 0
                current_controller = getattr(frame, f"{side}_controller")
                current_elbow = getattr(frame, f"{side}_elbow")
                accepted_controller = getattr(accepted, f"{side}_controller")
                accepted_elbow = getattr(accepted, f"{side}_elbow")
                hold_elbow = position_jump or elbow_jump
                if position_jump or rotation_jump:
                    replaces[f"{side}_controller"] = Pose(
                        self._reexpress_point(
                            accepted, frame, side, accepted_controller.position
                        ) if position_jump else current_controller.position,
                        self._reexpress_rotation(
                            accepted, frame, accepted_controller.rotation
                        ) if rotation_jump else current_controller.rotation,
                    )
                if hold_elbow:
                    replaces[f"{side}_elbow"] = Pose(
                        self._reexpress_point(accepted, frame, side, accepted_elbow.position),
                        self._reexpress_rotation(accepted, frame, accepted_elbow.rotation),
                    )
                if position_jump:
                    reasons.append(f"{side} controller position {position_delta:.2f} m")
                if rotation_jump:
                    reasons.append(f"{side} controller rotation {rotation_delta:.0f} deg")
                if elbow_jump:
                    reasons.append(f"{side} elbow {elbow_delta:.2f} m")
                continue
            self.sensor_stable_count[side] = self.sensor_stable_count.get(side, 0) + 1
            if self.sensor_stable_count[side] < 4:
                accepted_controller = getattr(accepted, f"{side}_controller")
                accepted_elbow = getattr(accepted, f"{side}_elbow")
                replaces[f"{side}_controller"] = Pose(
                    self._reexpress_point(accepted, frame, side, accepted_controller.position),
                    self._reexpress_rotation(accepted, frame, accepted_controller.rotation),
                )
                replaces[f"{side}_elbow"] = Pose(
                    self._reexpress_point(accepted, frame, side, accepted_elbow.position),
                    self._reexpress_rotation(accepted, frame, accepted_elbow.rotation),
                )
                continue
            gap_position, gap_rotation, gap_elbow = self._arm_deltas(frame, accepted, side)
            within_packet = (
                gap_position <= 0.15
                and gap_rotation <= 40.0
                and not (gap_elbow > 0.08 and gap_position < 0.06)
            )
            if within_packet:
                continue
            accepted_controller = getattr(accepted, f"{side}_controller")
            accepted_elbow = getattr(accepted, f"{side}_elbow")
            current_controller = getattr(frame, f"{side}_controller")
            current_elbow = getattr(frame, f"{side}_elbow")
            position, position_limited = self._slew_point(
                self._local_point(accepted, side, accepted_controller.position),
                self._local_point(frame, side, current_controller.position),
                position_budget,
            )
            rotation, rotation_limited = self._slew_rotation(
                self._local_rotation(accepted, accepted_controller.rotation),
                self._local_rotation(frame, current_controller.rotation),
                rotation_budget,
            )
            elbow_position, elbow_limited = self._slew_point(
                self._local_point(accepted, side, accepted_elbow.position),
                self._local_point(frame, side, current_elbow.position),
                elbow_budget,
            )
            elbow_rotation, elbow_rotation_limited = self._slew_rotation(
                self._local_rotation(accepted, accepted_elbow.rotation),
                self._local_rotation(frame, current_elbow.rotation),
                rotation_budget,
            )
            if position_limited or rotation_limited:
                replaces[f"{side}_controller"] = Pose(
                    self._sensor_anchor(frame, side) + frame.waist.rotation.apply(position),
                    frame.waist.rotation * rotation,
                )
            if elbow_limited or elbow_rotation_limited:
                replaces[f"{side}_elbow"] = Pose(
                    self._sensor_anchor(frame, side) + frame.waist.rotation.apply(elbow_position),
                    frame.waist.rotation * elbow_rotation,
                )
            if position_limited:
                reasons.append(f"{side} controller position {gap_position:.2f} m")
            if rotation_limited:
                reasons.append(f"{side} controller rotation {gap_rotation:.0f} deg")
            if elbow_limited:
                reasons.append(f"{side} elbow {gap_elbow:.2f} m")
        self.previous_raw_sensor = frame
        if not replaces:
            self.accepted_sensor = frame
            self.sensor_glitch = yaw_reason
            return frame
        if yaw_reason:
            reasons.insert(0, yaw_reason)
        cleaned = replace(frame, **replaces)
        self.accepted_sensor = cleaned
        self.sensor_glitch = ", ".join(reasons) if reasons else None
        return cleaned

    def map(self, frame):
        if not self.calibrated:
            raise RuntimeError("PoseArmMapper must be calibrated before mapping")
        head_position = self._filter_head_position(frame.head.position)
        wrist_positions = {
            side: self._filter_wrist_position(
                side, self._controller_wrist_world(frame, side)
            )
            for side in ("left", "right")
        }
        self._mark_calibration_pose(frame, wrist_positions)
        if self.at_calibration_pose and self.calibration_targets is not None:
            self.last_debug_metrics = {"mapping_pair_held": 0.0}
            self.last_targets = self.calibration_targets
            return self.calibration_targets
        if self.mode == "cooperative_wrist":
            return self._map_cooperative(frame, head_position, wrist_positions)
        self.last_debug_metrics = {}
        return DualArmTargets(
            left=self._map_arm(frame, "left", head_position, wrist_positions["left"]),
            right=self._map_arm(frame, "right", head_position, wrist_positions["right"]),
        )

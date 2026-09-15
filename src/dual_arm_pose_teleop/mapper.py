from dataclasses import dataclass

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
        self.last_debug_metrics = {}

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
        self.last_debug_metrics = {}

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
        return SensorFrame(
            head=cls._mean_pose([frame.head for frame in frames]),
            waist=cls._mean_pose([frame.waist for frame in frames]),
            left_elbow=cls._mean_pose([frame.left_elbow for frame in frames]),
            right_elbow=cls._mean_pose([frame.right_elbow for frame in frames]),
            left_controller=cls._mean_pose([frame.left_controller for frame in frames]),
            right_controller=cls._mean_pose([frame.right_controller for frame in frames]),
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
        """Reconstruct the elbow without using its runtime tracked position."""
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
            human_wrist_vector_local = waist_inv.apply(
                human_wrist_world - human_shoulder_world
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
            minimum = abs(arm.upper_length-arm.forearm_length)+self.reach_margin
            maximum = min((arm.upper_length+arm.forearm_length)*self.maximum_reach_ratio,
                          arm.upper_length+arm.forearm_length-self.reach_margin)
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
            raw_elbows[side] = (
                arm.shoulder
                + arm.upper_length * normalize(
                    arm.direction_alignment.apply(upper_h)
                )
            )
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
            elbow, wrist, elbow_direction = project_two_link(
                arm.shoulder,
                raw_elbows[side],
                coupled_wrists[side],
                arm.upper_length,
                arm.forearm_length,
                self.maximum_reach_ratio,
                self.reach_margin,
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
        if self.mode == "cooperative_wrist":
            return self._map_cooperative(frame, head_position, wrist_positions)
        self.last_debug_metrics = {}
        return DualArmTargets(
            left=self._map_arm(frame, "left", head_position, wrist_positions["left"]),
            right=self._map_arm(frame, "right", head_position, wrist_positions["right"]),
        )

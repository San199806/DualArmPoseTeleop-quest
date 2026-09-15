import numpy as np
import time
from scipy.spatial.transform import Rotation

from .types import Pose, SensorFrame


class XRobotInput:
    """XRobot adapter for head, waist, two elbow trackers and two controllers."""

    UNITY_TO_RIGHT_HANDED = Rotation.from_matrix(
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    )

    def __init__(self, tracker_config):
        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as error:
            raise RuntimeError("xrobotoolkit_sdk is required on the teleoperation PC") from error
        self.xrt = xrt
        self.trackers = tracker_config
        self.last_warning_time = 0.0
        self.pose_change_times = {}
        self.previous_poses = {}
        self.xrt.init()
        print("[XRobot] motion group timestamp: " +
              ("available" if hasattr(self.xrt, "get_motion_timestamp_ns") else "unavailable"))
        print("[XRobot] SDK has no verified per-tracker freshness API; unchanged poses are diagnostic only")

    def _warn(self, message):
        now = time.monotonic()
        if now - self.last_warning_time >= 1.0:
            print(f"[XRobot] {message}")
            self.last_warning_time = now

    def _convert_pose(self, raw_pose):
        if raw_pose is None or len(raw_pose) < 7:
            raise ValueError("Invalid XRobot pose")
        values = np.asarray(raw_pose[:7], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("XRobot pose contains NaN or infinity")
        quaternion = np.array(
            [values[6], values[3], values[4], values[5]], dtype=np.float64
        )
        if np.linalg.norm(quaternion) < 1e-8:
            raise ValueError("XRobot pose contains a zero quaternion")
        position = self.UNITY_TO_RIGHT_HANDED.apply(values[:3])
        raw_rotation = Rotation.from_quat(
            quaternion,
            scalar_first=True,
        )
        return Pose(position, self.UNITY_TO_RIGHT_HANDED * raw_rotation)

    def read(self):
        controller_data = self.controller_data()
        try:
            if hasattr(self.xrt, "get_motion_timestamp_ns"):
                controller_data["device_timestamps"] = {
                    "motion_group": int(self.xrt.get_motion_timestamp_ns())
                }
            serials = self.xrt.get_motion_tracker_serial_numbers()
            poses = self.xrt.get_motion_tracker_pose()
            serials = [] if serials is None else serials
            poses = [] if poses is None else poses
        except Exception as error:
            self._warn(f"tracker read failed: {error}")
            return None, controller_data
        if len(serials) != len(poses):
            self._warn(
                f"tracker serial/pose count mismatch: {len(serials)}/{len(poses)}"
            )
        motion = dict(zip(serials, poses))
        required = (
            self.trackers["waist"],
            self.trackers["left_elbow"],
            self.trackers["right_elbow"],
        )
        if not all(serial in motion for serial in required):
            missing = [serial for serial in required if serial not in motion]
            self._warn(f"required trackers missing: {missing}")
            return None, controller_data
        try:
            frame = SensorFrame(
                head=self._convert_pose(self.xrt.get_headset_pose()),
                waist=self._convert_pose(motion[self.trackers["waist"]]),
                left_elbow=self._convert_pose(motion[self.trackers["left_elbow"]]),
                right_elbow=self._convert_pose(motion[self.trackers["right_elbow"]]),
                left_controller=self._convert_pose(
                    self.xrt.get_left_controller_pose()
                ),
                right_controller=self._convert_pose(
                    self.xrt.get_right_controller_pose()
                ),
            )
        except Exception as error:
            self._warn(f"invalid pose frame: {error}")
            return None, controller_data
        now = time.monotonic()
        unchanged = {}
        for name in ("head", "waist", "left_elbow", "right_elbow", "left_controller", "right_controller"):
            pose = getattr(frame, name)
            values = np.r_[pose.position, pose.rotation.as_matrix().ravel()]
            previous = self.previous_poses.get(name)
            if previous is None or not np.array_equal(previous, values):
                self.pose_change_times[name] = now
            self.previous_poses[name] = values
            unchanged[name] = (now-self.pose_change_times[name])*1000
        controller_data["pose_unchanged_ms"] = unchanged
        return frame, controller_data

    @staticmethod
    def _plain(value):
        return value.tolist() if hasattr(value, "tolist") else value

    def close(self):
        if hasattr(self.xrt, "close"):
            self.xrt.close()

    def controller_data(self):
        try:
            return {
                "LeftController": {
                    "index_trig": self._plain(self.xrt.get_left_trigger()),
                    "grip": self._plain(self.xrt.get_left_grip()),
                    "key_one": bool(self.xrt.get_X_button()),
                    "key_two": bool(self.xrt.get_Y_button()),
                    "axis": self._plain(self.xrt.get_left_axis()),
                    "axis_click": bool(self.xrt.get_left_axis_click()),
                },
                "RightController": {
                    "index_trig": self._plain(self.xrt.get_right_trigger()),
                    "grip": self._plain(self.xrt.get_right_grip()),
                    "key_one": bool(self.xrt.get_A_button()),
                    "key_two": bool(self.xrt.get_B_button()),
                    "axis": self._plain(self.xrt.get_right_axis()),
                    "axis_click": bool(self.xrt.get_right_axis_click()),
                },
                "timestamp": int(self.xrt.get_time_stamp_ns()),
            }
        except Exception as error:
            self._warn(f"controller read failed: {error}")
            return {
                "LeftController": {
                    "index_trig": 0.0, "grip": 0.0,
                    "key_one": False, "key_two": False,
                    "axis": [0.0, 0.0], "axis_click": False,
                },
                "RightController": {
                    "index_trig": 0.0, "grip": 0.0,
                    "key_one": False, "key_two": False,
                    "axis": [0.0, 0.0], "axis_click": False,
                },
                "timestamp": None,
            }

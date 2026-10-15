"""Upper-body input from a Quest FullBodyTarget stream.

By default this process listens on ws://0.0.0.0:8000/api/move/ws/set_target,
the same endpoint Tools/mock_daemon_8000.py exposed. Passing a base URL polls
an external daemon instead. Bone positions are already robot FLU (x forward,
y left, z up). Quest controllers.left/right supply the same A/B/X/Y, trigger,
grip, and thumbstick values.
"""
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
from scipy.spatial.transform import Rotation

from .quest_target_server import QuestTargetServer
from .types import Pose, SensorFrame


_KEY_A = 65
_KEY_X = 88
_KEY_Y = 89


def _vector(position, label):
    if not isinstance(position, dict):
        raise ValueError(f"{label} position is missing")
    values = np.array([
        float(position.get("x", "nan")),
        float(position.get("y", "nan")),
        float(position.get("z", "nan")),
    ], dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} position is not finite")
    return values


def _quaternion(rotation, label):
    if not isinstance(rotation, dict):
        raise ValueError(f"{label} rotation is missing")
    values = np.array([
        float(rotation.get("x", "nan")),
        float(rotation.get("y", "nan")),
        float(rotation.get("z", "nan")),
        float(rotation.get("w", "nan")),
    ], dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not np.all(np.isfinite(values)) or norm < 1e-8:
        raise ValueError(f"{label} rotation is invalid")
    return Rotation.from_quat(values/norm)


def _joint_pose(joint, label):
    if not isinstance(joint, dict):
        raise ValueError(f"{label} joint is missing")
    return Pose(_vector(joint.get("position"), label), _quaternion(joint.get("rotation"), label))


def _upper_arm_rotation(direction):
    """Rotation whose +Y axis is the upper arm, matching upper_arm_axis '-y'."""
    y_axis = np.asarray(direction, dtype=np.float64)
    length = float(np.linalg.norm(y_axis))
    if length < 1e-6:
        raise ValueError("Upper arm length is too small")
    y_axis = y_axis/length
    auxiliary = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(y_axis, auxiliary))) > 0.9:
        auxiliary = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(y_axis, auxiliary)
    x_axis = x_axis/np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    return Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis]))


def _head_rotation(payload):
    pose = payload.get("target_head_pose") or {}
    matrix = pose.get("m") if isinstance(pose, dict) else None
    if not isinstance(matrix, list) or len(matrix) != 16:
        return Rotation.identity()
    values = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(values)):
        raise ValueError("target_head_pose contains a non-finite value")
    rotation = values[:3, :3]
    if abs(float(np.linalg.det(rotation))) < 1e-6:
        raise ValueError("target_head_pose rotation is degenerate")
    return Rotation.from_matrix(rotation)


def _body_yaw(payload, left_shoulder, right_shoulder):
    """Yaw of the shoulder line. target_body_yaw is only a fallback.

    The Quest app clamps target_body_yaw to ±45° around the tracking
    forward, so a calibration that is already turned leaves one direction
    with almost no travel.
    """
    lateral = left_shoulder-right_shoulder
    if float(np.linalg.norm(lateral[:2])) >= 1e-4:
        return float(np.arctan2(-lateral[0], lateral[1]))
    if payload.get("target_body_yaw") is not None:
        yaw = float(payload["target_body_yaw"])
        if not np.isfinite(yaw):
            raise ValueError("target_body_yaw is not finite")
        return yaw
    raise ValueError("Shoulders do not define a body yaw")


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _hand(payload, side):
    controllers = payload.get("controllers") if isinstance(payload, dict) else None
    hand = controllers.get(side) if isinstance(controllers, dict) else None
    return hand if isinstance(hand, dict) else {}


def _stick(hand):
    stick = hand.get("thumbstick")
    if not isinstance(stick, dict):
        return [0.0, 0.0]
    return [_number(stick.get("x")), _number(stick.get("y"))]


def frame_from_target(payload):
    """Build a SensorFrame from one FullBodyTarget or /api/state/full_state body.

    Head position is the shoulder midpoint. The daemon head matrix is a yawed
    orientation command and its translation is often zero, while the arm bones
    are in the tracking frame. Waist yaw follows the shoulder line. The elbow pose
    sits on the lower-arm bone and points +Y from shoulder to elbow. A controller
    pose, when the Quest app sends one, replaces the body wrist bone.
    """
    if not isinstance(payload, dict) or payload.get("status") == "no_quest_data_yet":
        return None
    arms = payload.get("arms")
    if not isinstance(arms, dict):
        return None
    parsed = {}
    for side in ("left", "right"):
        side_data = arms.get(side)
        if not isinstance(side_data, dict):
            return None
        try:
            parsed[side] = {
                name: _joint_pose(side_data.get(name), f"{side} {name}")
                for name in ("shoulder", "lower_arm", "wrist")
            }
        except (TypeError, ValueError):
            return None
    left_shoulder = parsed["left"]["shoulder"].position
    right_shoulder = parsed["right"]["shoulder"].position
    head_position = 0.5*(left_shoulder+right_shoulder)
    try:
        yaw = _body_yaw(payload, left_shoulder, right_shoulder)
        head_rotation = _head_rotation(payload)
    except (TypeError, ValueError):
        return None
    waist = Pose(head_position.copy(), Rotation.from_euler("z", yaw))
    elbows = {}
    for side in ("left", "right"):
        shoulder = parsed[side]["shoulder"].position
        elbow = parsed[side]["lower_arm"].position
        try:
            rotation = _upper_arm_rotation(elbow-shoulder)
        except ValueError:
            return None
        elbows[side] = Pose(elbow.copy(), rotation)
    return SensorFrame(
        head=Pose(head_position.copy(), head_rotation),
        waist=waist,
        left_elbow=elbows["left"],
        right_elbow=elbows["right"],
        left_controller=_controller_pose(payload, "left") or parsed["left"]["wrist"],
        right_controller=_controller_pose(payload, "right") or parsed["right"]["wrist"],
        left_shoulder=parsed["left"]["shoulder"],
        right_shoulder=parsed["right"]["shoulder"],
        hips=_named_pose(payload, "hips"),
        chest=_named_pose(payload, "chest"),
    )


def _named_pose(payload, name):
    raw = payload.get(name)
    if not isinstance(raw, dict):
        return None
    try:
        return _joint_pose(raw, name)
    except (TypeError, ValueError):
        return None


def _controller_pose(payload, side):
    """Grip pose from controllers.left/right. Missing fields keep the wrist bone."""
    hand = _hand(payload, side)
    if hand.get("connected") is False:
        return None
    if "position" not in hand and "rotation" not in hand:
        return None
    try:
        return _joint_pose(hand, f"{side} controller")
    except (TypeError, ValueError):
        return None


class SkeletonInput:
    """Expose the latest upper body as a SensorFrame.

    With no base URL, Quest connects directly to this process. A base URL
    keeps the older HTTP poll of an external daemon.
    """

    def __init__(self, base_url=None, *, host="0.0.0.0", port=8000, key_hold_s=1.25, stream_hold_s=0.5):
        self.key_hold_s = float(key_hold_s)
        self.stream_hold_s = float(stream_hold_s)
        self._lock = threading.Lock()
        self._payload = None
        self._message_count = 0
        self._pulse = set()
        self._y_until = 0.0
        self._stop = threading.Event()
        self._warned_at = 0.0
        self._seen_frame = False
        self._stream_lost = False
        self._quest_connected = False
        self._rate_window_at = None
        self._rate_window_count = 0
        self.previous_poses = {}
        self.pose_change_times = {}
        self._thread = None
        self._server = None
        if base_url:
            self.endpoint = self._endpoint(base_url)
            self._thread = threading.Thread(target=self._poll, name="skeleton-input", daemon=True)
            self._thread.start()
            print(f"[Input] upper-body skeleton {self.endpoint}")
        else:
            self.endpoint = None
            self._server = QuestTargetServer(self._on_quest_payload, host=host, port=port).start()
            print(
                "[Input] Quest skeleton "
                f"ws://{host}:{self._server.port}/api/move/ws/set_target"
            )

    @staticmethod
    def _endpoint(base_url):
        url = str(base_url).rstrip("/")
        if "full_state" in url:
            return url
        return (url+"/api/state/full_state"
                "?with_target_head_pose=true&with_target_body_yaw=true")

    def _warn(self, message):
        now = time.monotonic()
        if now-self._warned_at >= 1.0:
            print(f"[Skeleton] {message}")
            self._warned_at = now

    def _poll(self):
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(self.endpoint, timeout=0.2) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("daemon state is not an object")
                with self._lock:
                    self._payload = payload
            except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError, TimeoutError) as error:
                self._warn(f"daemon read failed: {error}")
            self._stop.wait(0.03)

    def _on_quest_payload(self, payload):
        """Store one Quest target, or return the HTTP full_state view when payload is None."""
        if payload is None:
            with self._lock:
                current = None if self._payload is None else dict(self._payload)
                count = self._message_count
            return _full_state(current, count)
        if not isinstance(payload, dict):
            raise ValueError("target is not an object")
        stored = dict(payload)
        stored["last_at"] = time.time()
        now = time.monotonic()
        with self._lock:
            self._payload = stored
            self._message_count += 1
            first = not self._quest_connected
            self._quest_connected = True
            count = self._message_count
            started = self._rate_window_at
            if started is None:
                self._rate_window_at = now
                self._rate_window_count = count
                rate_hz = None
            elif now-started >= 1.0:
                rate_hz = (count-self._rate_window_count)/(now-started)
                self._rate_window_at = now
                self._rate_window_count = count
            else:
                rate_hz = None
        if first:
            print("[Skeleton] Quest connected")
        if rate_hz is not None:
            print(f"[Skeleton] Quest packets {rate_hz:.1f} Hz")
        return None

    def press_key(self, keycode):
        """Viewer key press. A and X are one-frame edges; Y stays down long enough to recalibrate."""
        name = {_KEY_A: "a", _KEY_X: "x", _KEY_Y: "y"}.get(int(keycode))
        if name is None:
            return
        with self._lock:
            if name == "y":
                self._y_until = time.monotonic()+self.key_hold_s
            else:
                self._pulse.add(name)

    def _controller(self, payload):
        """Quest primary is X/A and secondary is Y/B. Viewer keys fill the same buttons."""
        left = _hand(payload, "left")
        right = _hand(payload, "right")
        with self._lock:
            pulsed = set(self._pulse)
            self._pulse.clear()
            y_held = time.monotonic() < self._y_until
        return {
            "LeftController": {
                "index_trig": _number(left.get("trigger")),
                "grip": _number(left.get("grip")),
                "key_one": bool(left.get("primary")) or "x" in pulsed,
                "key_two": bool(left.get("secondary")) or y_held,
                "axis": _stick(left),
                "axis_click": bool(left.get("thumbstick_click")),
            },
            "RightController": {
                "index_trig": _number(right.get("trigger")),
                "grip": _number(right.get("grip")),
                "key_one": bool(right.get("primary")) or "a" in pulsed,
                "key_two": bool(right.get("secondary")),
                "axis": _stick(right),
                "axis_click": bool(right.get("thumbstick_click")),
            },
            "timestamp": None,
        }

    def _live_payload(self):
        with self._lock:
            payload = None if self._payload is None else dict(self._payload)
        if payload is None:
            return None
        last_at = payload.get("last_at")
        if last_at is None:
            return payload
        try:
            age = time.time()-float(last_at)
        except (TypeError, ValueError):
            return payload
        if age > self.stream_hold_s:
            self._warn(f"Quest skeleton is {age*1000:.0f} ms old")
            return None
        return payload

    def read(self):
        payload = self._live_payload()
        controller = self._controller(payload or {})
        frame = frame_from_target(payload) if payload is not None else None
        if frame is None:
            if self._seen_frame and not self._stream_lost:
                print("[Skeleton] Quest skeleton stopped")
                self._stream_lost = True
            if payload is not None and payload.get("status") != "no_quest_data_yet":
                self._warn("upper-body skeleton is incomplete")
            return None, controller
        controller["timestamp"] = time.time_ns()
        if self._stream_lost:
            print("[Skeleton] Quest skeleton resumed")
            self._stream_lost = False
        if not self._seen_frame:
            self._seen_frame = True
            print("[Skeleton] receiving upper-body skeleton")
        now = time.monotonic()
        unchanged = {}
        for name in ("head", "waist", "left_elbow", "right_elbow", "left_controller", "right_controller"):
            pose = getattr(frame, name)
            values = np.r_[pose.position, pose.rotation.as_quat()]
            previous = self.previous_poses.get(name)
            if previous is None or not np.array_equal(previous, values):
                self.pose_change_times[name] = now
            self.previous_poses[name] = values
            unchanged[name] = (now-self.pose_change_times[name])*1000
        controller["pose_unchanged_ms"] = unchanged
        return frame, controller

    def close(self):
        self._stop.set()
        if self._server is not None:
            self._server.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _full_state(payload, count):
    out = {"mock": True, "message_count": count, "last_at": 0.0 if payload is None else payload.get("last_at", 0.0)}
    if payload is None:
        out["status"] = "no_quest_data_yet"
        return out
    for name in ("target_head_pose", "target_body_yaw", "arms", "controllers"):
        if name in payload:
            out[name] = payload[name]
    return out

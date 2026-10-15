import base64
import json
import socket
import struct
import threading
import time
import unittest

import numpy as np

from dual_arm_pose_teleop.mapper import PoseArmMapper
from dual_arm_pose_teleop.quest_target_server import accept_key
from dual_arm_pose_teleop.skeleton_input import SkeletonInput, frame_from_target


SAMPLE = {
    "target_head_pose": {"m": [
        0.9197384, 0.0585792, 0.388136268, 0.0,
        -0.09144562, 0.9935712, 0.0667380244, 0.0,
        -0.38173157, -0.0968748853, 0.9191824, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]},
    "target_antennas": [0.360718966, -1.022241],
    "target_body_yaw": -0.0270831864,
    "arms": {
        "left": {
            "shoulder": {
                "position": {"x": -0.0795345, "y": 0.02147773, "z": 0.937897265},
                "rotation": {"x": -0.161652833, "y": 0.7609229, "z": 0.1559256, "w": 0.608729839},
            },
            "upper_arm": {
                "position": {"x": -0.165002376, "y": 0.168879449, "z": 0.955693662},
                "rotation": {"x": -0.243288055, "y": 0.8253461, "z": -0.232364178, "w": 0.45345515},
            },
            "lower_arm": {
                "position": {"x": -0.214366645, "y": 0.367482364, "z": 0.8008541},
                "rotation": {"x": -0.235314757, "y": -0.277030051, "z": 0.8114455, "w": -0.457643569},
            },
            "wrist": {
                "position": {"x": -0.000442236662, "y": 0.2713319, "z": 0.7582547},
                "rotation": {"x": -0.5361347, "y": -0.308632046, "z": 0.784279644, "w": -0.04702397},
            },
        },
        "right": {
            "shoulder": {
                "position": {"x": -0.0810039937, "y": -0.0327675045, "z": 0.937343061},
                "rotation": {"x": -0.162842169, "y": -0.6051225, "z": -0.180312008, "w": 0.758153558},
            },
            "upper_arm": {
                "position": {"x": -0.17152667, "y": -0.173702836, "z": 0.951765537},
                "rotation": {"x": 0.450380027, "y": -0.471773744, "z": -0.357281, "w": 0.6685339},
            },
            "lower_arm": {
                "position": {"x": -0.1854867, "y": -0.260818362, "z": 0.7107898},
                "rotation": {"x": 0.8036989, "y": -0.572991, "z": 0.147991627, "w": 0.0620323},
            },
            "wrist": {
                "position": {"x": 0.0394828022, "y": -0.1883406, "z": 0.7398163},
                "rotation": {"x": 0.678711832, "y": -0.3686001, "z": 0.39217034, "w": 0.499686658},
            },
        },
    },
}


class SkeletonInputTests(unittest.TestCase):
    def test_sample_maps_bones_onto_the_retarget_frame(self):
        frame = frame_from_target(dict(SAMPLE, last_at=1.0))
        left_shoulder = np.array([-0.0795345, 0.02147773, 0.937897265])
        left_elbow = np.array([-0.214366645, 0.367482364, 0.8008541])
        left_wrist = np.array([-0.000442236662, 0.2713319, 0.7582547])
        right_shoulder = np.array([-0.0810039937, -0.0327675045, 0.937343061])
        np.testing.assert_allclose(frame.left_elbow.position, left_elbow)
        np.testing.assert_allclose(frame.left_controller.position, left_wrist)
        np.testing.assert_allclose(frame.head.position, 0.5*(left_shoulder+right_shoulder))
        upper = left_elbow-left_shoulder
        upper = upper/np.linalg.norm(upper)
        np.testing.assert_allclose(frame.left_elbow.rotation.apply([0, 1, 0]), upper)
        lateral = left_shoulder-right_shoulder
        np.testing.assert_allclose(
            frame.waist.rotation.as_rotvec(),
            [0, 0, np.arctan2(-lateral[0], lateral[1])],
        )
        self.assertAlmostEqual(np.linalg.det(frame.head.rotation.as_matrix()), 1.0, places=5)

    def test_controller_pose_replaces_the_body_wrist(self):
        payload = json.loads(json.dumps(SAMPLE))
        payload["controllers"] = {
            "left": {
                "connected": True,
                "position": {"x": 0.4, "y": 0.2, "z": 1.1},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "right": {
                "connected": True,
                "position": {"x": 0.4, "y": -0.2, "z": 1.1},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        }
        frame = frame_from_target(payload)
        np.testing.assert_allclose(frame.left_controller.position, [0.4, 0.2, 1.1])
        np.testing.assert_allclose(frame.right_controller.position, [0.4, -0.2, 1.1])
        np.testing.assert_allclose(
            frame.left_elbow.position,
            [-0.214366645, 0.367482364, 0.8008541],
        )

    def test_missing_skeleton_is_not_a_frame(self):
        self.assertIsNone(frame_from_target({"status": "no_quest_data_yet"}))
        self.assertIsNone(frame_from_target({"target_body_yaw": 0.0}))

    def test_shoulder_line_is_not_clamped_to_the_quest_yaw_limit(self):
        payload = json.loads(json.dumps(SAMPLE))
        payload["target_body_yaw"] = 0.785398
        payload["arms"]["left"]["shoulder"]["position"] = {"x": -0.2, "y": 0.0, "z": 1.0}
        payload["arms"]["right"]["shoulder"]["position"] = {"x": 0.2, "y": 0.0, "z": 1.0}
        frame = frame_from_target(payload)
        self.assertAlmostEqual(frame.waist.rotation.as_rotvec()[2], np.pi/2, places=5)

    def test_hips_and_chest_are_kept_for_torso_pitch(self):
        payload = json.loads(json.dumps(SAMPLE))
        identity = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        payload["hips"] = {"position": {"x": 0.0, "y": 0.0, "z": 0.9}, "rotation": identity}
        payload["chest"] = {"position": {"x": 0.2, "y": 0.0, "z": 1.3}, "rotation": identity}
        frame = frame_from_target(payload)
        self.assertGreater(
            PoseArmMapper.torso_pitch(frame.hips.position, frame.chest.position, 0.0),
            0.0,
        )

    def test_quest_controllers_drive_the_same_buttons(self):
        adapter = SkeletonInput.__new__(SkeletonInput)
        adapter._lock = threading.Lock()
        adapter._pulse = set()
        adapter._y_until = 0.0
        adapter.key_hold_s = 1.25
        payload = {"controllers": {
            "left": {
                "connected": True, "trigger": 0.4, "grip": 0.2,
                "primary": True, "secondary": True, "thumbstick_click": True,
                "menu": False, "thumbstick": {"x": -0.3, "y": 0.5},
            },
            "right": {
                "connected": True, "trigger": 0.8, "grip": 0.1,
                "primary": True, "secondary": False, "thumbstick_click": False,
                "menu": False, "thumbstick": {"x": 0.2, "y": -0.7},
            },
        }}
        controller = adapter._controller(payload)
        left = controller["LeftController"]
        right = controller["RightController"]
        self.assertTrue(left["key_one"])
        self.assertTrue(left["key_two"])
        self.assertAlmostEqual(left["index_trig"], 0.4)
        self.assertAlmostEqual(left["grip"], 0.2)
        self.assertEqual(left["axis"], [-0.3, 0.5])
        self.assertTrue(left["axis_click"])
        self.assertTrue(right["key_one"])
        self.assertFalse(right["key_two"])
        self.assertAlmostEqual(right["index_trig"], 0.8)
        self.assertEqual(right["axis"], [0.2, -0.7])
        held = adapter._controller(payload)
        self.assertTrue(held["LeftController"]["key_two"])
        self.assertTrue(held["RightController"]["key_one"])

    def test_released_quest_controllers_leave_buttons_up(self):
        adapter = SkeletonInput.__new__(SkeletonInput)
        adapter._lock = threading.Lock()
        adapter._pulse = set()
        adapter._y_until = 0.0
        adapter.key_hold_s = 1.25
        payload = {"controllers": {
            "left": {
                "connected": False, "trigger": 0.0, "grip": 0.0,
                "primary": False, "secondary": False, "thumbstick_click": False,
                "menu": False, "thumbstick": {"x": 0.0, "y": 0.0},
            },
            "right": {
                "connected": True, "trigger": 0.0, "grip": 0.0,
                "primary": False, "secondary": False, "thumbstick_click": False,
                "menu": False, "thumbstick": {"x": 0.0, "y": 0.0},
            },
        }}
        controller = adapter._controller(payload)
        self.assertFalse(controller["LeftController"]["key_one"])
        self.assertFalse(controller["LeftController"]["key_two"])
        self.assertFalse(controller["RightController"]["key_one"])
        self.assertFalse(controller["RightController"]["key_two"])

    def test_keyboard_edges_match_controller_buttons(self):
        adapter = SkeletonInput.__new__(SkeletonInput)
        adapter._lock = threading.Lock()
        adapter._pulse = set()
        adapter._y_until = 0.0
        adapter.key_hold_s = 1.25
        adapter.press_key(65)
        controller = adapter._controller({})
        self.assertTrue(controller["RightController"]["key_one"])
        self.assertFalse(adapter._controller({})["RightController"]["key_one"])
        adapter.press_key(89)
        self.assertTrue(adapter._controller({})["LeftController"]["key_two"])


    def test_cached_quest_pose_stays_live_between_packets(self):
        adapter = SkeletonInput.__new__(SkeletonInput)
        adapter._lock = threading.Lock()
        adapter._pulse = set()
        adapter._y_until = 0.0
        adapter.key_hold_s = 1.25
        adapter.stream_hold_s = 0.5
        adapter._warned_at = 0.0
        adapter._seen_frame = True
        adapter._stream_lost = False
        adapter.previous_poses = {}
        adapter.pose_change_times = {}
        adapter._payload = dict(SAMPLE, last_at=time.time()-0.15)
        frame, controller = adapter.read()
        self.assertIsNotNone(frame)
        self.assertIsNotNone(controller["timestamp"])
        adapter._payload = dict(SAMPLE, last_at=time.time()-0.6)
        frame, controller = adapter.read()
        self.assertIsNone(frame)
        self.assertIsNone(controller["timestamp"])

    def test_embedded_server_receives_a_quest_target(self):
        self.assertEqual(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")
        adapter = SkeletonInput(host="127.0.0.1", port=0)
        self.addCleanup(adapter.close)
        payload = json.loads(json.dumps(SAMPLE))
        payload["controllers"] = {
            "right": {
                "connected": True, "trigger": 0.0, "grip": 0.0,
                "primary": True, "secondary": False, "thumbstick_click": False,
                "menu": False, "thumbstick": {"x": 0.0, "y": 0.0},
            },
        }
        sock = _connect_quest(adapter._server.port)
        self.addCleanup(sock.close)
        sock.sendall(_client_text(json.dumps(payload)))
        frame = None
        controller = None
        deadline = time.time()+2
        while time.time() < deadline:
            frame, controller = adapter.read()
            if frame is not None:
                break
            time.sleep(0.01)
        self.assertAlmostEqual(frame.left_controller.position[1], 0.2713319)
        self.assertTrue(controller["RightController"]["key_one"])
        self.assertIsNotNone(controller["timestamp"])

        state_sock = socket.create_connection(("127.0.0.1", adapter._server.port), timeout=2)
        self.addCleanup(state_sock.close)
        state_sock.sendall(b"GET /api/state/full_state HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        state = json.loads(_read_http_body(state_sock))
        self.assertEqual(state["message_count"], 1)
        self.assertIn("arms", state)

        sock.sendall(_client_text("not-json"))
        error = json.loads(_read_server_text(sock))
        self.assertEqual(error["status"], "error")


def _connect_quest(port):
    sock = socket.create_connection(("127.0.0.1", port), timeout=2)
    key = base64.b64encode(b"0123456789abcdef").decode("ascii")
    request = (
        "GET /api/move/ws/set_target HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    )
    sock.sendall(request.encode("ascii"))
    data = b""
    while b"\r\n\r\n" not in data:
        data += sock.recv(4096)
    self_status = data.split(b"\r\n", 1)[0]
    if b" 101 " not in self_status:
        raise AssertionError(data.decode("iso-8859-1", "replace"))
    return sock


def _client_text(text):
    data = text.encode("utf-8")
    mask = b"\x11\x22\x33\x44"
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(data))
    length = len(data)
    if length < 126:
        header = bytes([0x81, 0x80 | length])
    else:
        header = bytes([0x81, 126 | 0x80])+struct.pack("!H", length)
    return header+mask+masked


def _read_exact(sock, size):
    data = b""
    while len(data) < size:
        chunk = sock.recv(size-len(data))
        if not chunk:
            raise ConnectionError("closed")
        data += chunk
    return data


def _read_server_text(sock):
    header = _read_exact(sock, 2)
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", _read_exact(sock, 2))[0]
    payload = _read_exact(sock, length)
    return payload.decode("utf-8")


def _read_http_body(sock):
    data = b""
    while b"\r\n\r\n" not in data:
        data += sock.recv(4096)
    head, body = data.split(b"\r\n\r\n", 1)
    for line in head.decode("iso-8859-1").split("\r\n"):
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1])
            while len(body) < length:
                body += sock.recv(length-len(body))
            return body.decode("utf-8")
    raise AssertionError(head)


if __name__ == "__main__":
    unittest.main()

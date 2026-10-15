"""Bounded asynchronous sensor recording and timed, offline replay."""
import json
import pathlib
import queue
import threading
import time
from dataclasses import fields

import numpy as np
from scipy.spatial.transform import Rotation
from .types import Pose, SensorFrame


def encode_frame(frame):
    if frame is None:
        return None
    encoded = {}
    for field in fields(SensorFrame):
        pose = getattr(frame, field.name)
        if pose is None:
            encoded[field.name] = None
            continue
        encoded[field.name] = {
            "position": pose.position.tolist(),
            "quaternion_xyzw": pose.rotation.as_quat().tolist(),
        }
    return encoded


def decode_frame(data):
    if data is None:
        return None
    poses = {}
    for field in fields(SensorFrame):
        item = data.get(field.name)
        if item is None:
            if field.name.endswith("shoulder") or field.name in ("hips", "chest"):
                poses[field.name] = None
                continue
            raise ValueError("Invalid recorded pose")
        position = np.asarray(item["position"], dtype=float)
        quaternion = np.asarray(item["quaternion_xyzw"], dtype=float)
        if position.shape != (3,) or quaternion.shape != (4,) or not np.all(np.isfinite(np.r_[position, quaternion])):
            raise ValueError("Invalid recorded pose")
        poses[field.name] = Pose(position, Rotation.from_quat(quaternion))
    return SensorFrame(**poses)


class RecordingInput:
    def __init__(self, source, path):
        self.source = source
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("x", encoding="utf-8")
        self.queue = queue.Queue(maxsize=2000)
        self.started = time.monotonic()
        self.dropped = 0
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            while True:
                sample = self.queue.get()
                if sample is None:
                    break
                self.file.write(json.dumps(sample, allow_nan=False)+"\n")
        except Exception as error:
            self.error = str(error)
        finally:
            self.file.close()

    def press_key(self, keycode):
        if hasattr(self.source, "press_key"):
            self.source.press_key(keycode)

    def read(self):
        frame, controller = self.source.read()
        sample = {"version": 1, "elapsed_s": time.monotonic()-self.started,
                  "frame": encode_frame(frame), "controller_data": controller}
        try:
            self.queue.put_nowait(sample)
        except queue.Full:
            self.dropped += 1
        return frame, controller

    @property
    def finished(self):
        return getattr(self.source, "finished", False)

    def close(self):
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=5)
        print(f"[Record] dropped={self.dropped}, error={self.error}")
        if hasattr(self.source, "close"):
            self.source.close()


class RecordedInput:
    def __init__(self, path):
        self.file = pathlib.Path(path).open(encoding="utf-8")
        self.started = time.monotonic()
        self.finished = False
        self.current = None
        self.next = self._read_sample()
        self.offset = self.next["elapsed_s"] if self.next else 0

    def _read_sample(self):
        line = self.file.readline()
        if not line:
            return None
        data = json.loads(line)
        if data.get("version") != 1 or not np.isfinite(data["elapsed_s"]):
            raise ValueError("Unsupported or invalid recording")
        return data

    def read(self):
        elapsed = time.monotonic()-self.started+self.offset
        while self.next is not None and self.next["elapsed_s"] <= elapsed:
            self.current = self.next
            self.next = self._read_sample()
        if self.current is None:
            self.finished = self.next is None
            return None, {}
        data = self.current
        if self.next is None:
            # Deliver the last frame once; terminate on the following read.
            self.current = None
        return decode_frame(data["frame"]), data["controller_data"]

    def close(self):
        self.file.close()

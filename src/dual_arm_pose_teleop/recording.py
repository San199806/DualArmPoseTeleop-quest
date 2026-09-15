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
    return {f.name: {"position": getattr(frame, f.name).position.tolist(),
                     "quaternion_xyzw": getattr(frame, f.name).rotation.as_quat().tolist()}
            for f in fields(SensorFrame)}


def decode_frame(data):
    if data is None:
        return None
    poses = {}
    for f in fields(SensorFrame):
        p = np.asarray(data[f.name]["position"], dtype=float)
        q = np.asarray(data[f.name]["quaternion_xyzw"], dtype=float)
        if p.shape != (3,) or q.shape != (4,) or not np.all(np.isfinite(np.r_[p, q])):
            raise ValueError("Invalid recorded pose")
        poses[f.name] = Pose(p, Rotation.from_quat(q))
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

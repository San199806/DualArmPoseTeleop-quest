import json
import pathlib
import time
from collections import deque

import numpy as np


class RuntimeDiagnostics:
    """Low-overhead timing and discontinuity diagnostics with a history buffer."""

    def __init__(self, robot_name, period_s, project_root, config=None):
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.period_ms = period_s * 1000.0
        self.summary_interval_s = float(config.get("summary_interval_s", 1.0))
        self.console_cooldown_s = float(config.get("console_cooldown_s", 0.5))
        self.event_cooldown_s = float(config.get("event_cooldown_s", 0.5))
        self.thresholds = {
            "loop_overrun_ms": float(
                config.get("loop_overrun_ms", max(15.0, self.period_ms * 1.5))
            ),
            "input_stall_ms": float(config.get("input_stall_ms", 15.0)),
            "redis_stall_ms": float(config.get("redis_stall_ms", 10.0)),
            "ik_stall_ms": float(config.get("ik_stall_ms", 15.0)),
            "sensor_gap_ms": float(config.get("sensor_gap_ms", 30.0)),
            "sensor_stale_ms": float(config.get("sensor_stale_ms", 100.0)),
            "sensor_position_jump_m": float(
                config.get("sensor_position_jump_m", 0.08)
            ),
            "sensor_rotation_jump_deg": float(
                config.get("sensor_rotation_jump_deg", 25.0)
            ),
            "target_position_jump_m": float(
                config.get("target_position_jump_m", 0.08)
            ),
            "target_rotation_jump_deg": float(
                config.get("target_rotation_jump_deg", 25.0)
            ),
            "joint_jump_rad": float(config.get("joint_jump_rad", 0.15)),
            "real_state_jump_rad": float(config.get("real_state_jump_rad", 0.15)),
        }
        self.history = deque(maxlen=max(10, int(config.get("history_frames", 100))))
        self.previous_frame = None
        self.previous_targets = None
        self.previous_command = None
        self.previous_real_state = None
        self.previous_sensor_timestamp = None
        self.sensor_timestamp_changed_at = time.perf_counter()
        self.previous_step_started = None
        self.last_summary = time.perf_counter()
        self.last_console = 0.0
        self.last_event = 0.0
        self.interval_count = 0
        self.interval_sums = {}
        self.interval_maxima = {}
        self.file = None
        self.path = None

        if self.enabled:
            log_dir = pathlib.Path(config.get("log_dir", "logs"))
            if not log_dir.is_absolute():
                log_dir = pathlib.Path(project_root) / log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.path = log_dir / f"teleop_{robot_name}_{stamp}.jsonl"
            self.file = self.path.open("a", encoding="utf-8", buffering=1)
            self._write(
                {
                    "type": "session",
                    "time": time.time(),
                    "robot": robot_name,
                    "period_ms": self.period_ms,
                    "thresholds": self.thresholds,
                }
            )
            print(f"[Diag] log: {self.path}")

    def begin_step(self):
        started = time.perf_counter()
        loop_dt_ms = None
        if self.previous_step_started is not None:
            loop_dt_ms = (started - self.previous_step_started) * 1000.0
        self.previous_step_started = started
        return started, loop_dt_ms

    @staticmethod
    def _rotation_delta_deg(previous, current):
        return float(np.degrees((previous.inv() * current).magnitude()))

    @staticmethod
    def _position_delta(previous, current):
        return float(np.linalg.norm(current - previous))

    def _sensor_deltas(self, frame):
        metrics = {}
        if frame is not None and self.previous_frame is not None:
            for name in (
                "head",
                "waist",
                "left_elbow",
                "right_elbow",
                "left_controller",
                "right_controller",
            ):
                previous = getattr(self.previous_frame, name)
                current = getattr(frame, name)
                metrics[f"sensor_{name}_position_jump_m"] = self._position_delta(
                    previous.position, current.position
                )
                metrics[f"sensor_{name}_rotation_jump_deg"] = (
                    self._rotation_delta_deg(previous.rotation, current.rotation)
                )
        if frame is not None:
            self.previous_frame = frame
        return metrics

    def _target_deltas(self, targets):
        metrics = {}
        if targets is not None and self.previous_targets is not None:
            for side in ("left", "right"):
                previous = getattr(self.previous_targets, side)
                current = getattr(targets, side)
                metrics[f"target_{side}_elbow_jump_m"] = self._position_delta(
                    previous.elbow.position, current.elbow.position
                )
                metrics[f"target_{side}_wrist_jump_m"] = self._position_delta(
                    previous.wrist.position, current.wrist.position
                )
                metrics[f"target_{side}_wrist_rotation_jump_deg"] = (
                    self._rotation_delta_deg(
                        previous.wrist.rotation, current.wrist.rotation
                    )
                )
        if targets is not None:
            self.previous_targets = targets
        return metrics

    def _array_jump(self, name, current, previous):
        if current is None:
            return {}, previous
        current = np.asarray(current, dtype=np.float64).copy()
        metrics = {}
        if previous is not None and current.shape == previous.shape:
            difference = np.abs(current - previous)
            metrics[name] = float(np.max(difference)) if difference.size else 0.0
        return metrics, current

    def _sensor_timestamp_gap(self, controller_data):
        timestamp = controller_data.get("timestamp")
        if timestamp is None:
            return {}
        timestamp = int(timestamp)
        metrics = {}
        now = time.perf_counter()
        if self.previous_sensor_timestamp is not None and timestamp != self.previous_sensor_timestamp:
            metrics["sensor_timestamp_gap_ms"] = (
                timestamp - self.previous_sensor_timestamp
            ) / 1e6
        if timestamp != self.previous_sensor_timestamp:
            self.sensor_timestamp_changed_at = now
        metrics["sensor_stale_ms"] = (
            now - self.sensor_timestamp_changed_at
        ) * 1000.0
        self.previous_sensor_timestamp = timestamp
        return metrics

    def _anomaly_reasons(self, metrics):
        reasons = []
        direct = {
            "total_ms": "loop_overrun_ms",
            "input_ms": "input_stall_ms",
            "ik_ms": "ik_stall_ms",
            "sensor_timestamp_gap_ms": "sensor_gap_ms",
            "sensor_stale_ms": "sensor_stale_ms",
            "tracking_stale_ms": "sensor_stale_ms",
            "command_joint_jump_rad": "joint_jump_rad",
            "real_state_jump_rad": "real_state_jump_rad",
        }
        for metric, threshold in direct.items():
            if metrics.get(metric, 0.0) > self.thresholds[threshold]:
                reasons.append(metric)
        if max(metrics.get("redis_read_ms", 0.0), metrics.get("redis_publish_ms", 0.0)) > self.thresholds["redis_stall_ms"]:
            reasons.append("redis_stall")
        for name, value in metrics.items():
            if name.startswith("sensor_") and name.endswith("position_jump_m"):
                if value > self.thresholds["sensor_position_jump_m"]:
                    reasons.append(name)
            elif name.startswith("sensor_") and name.endswith("rotation_jump_deg"):
                if value > self.thresholds["sensor_rotation_jump_deg"]:
                    reasons.append(name)
            elif name.startswith("target_") and name.endswith("jump_m"):
                if value > self.thresholds["target_position_jump_m"]:
                    reasons.append(name)
            elif name.startswith("target_") and name.endswith("rotation_jump_deg"):
                if value > self.thresholds["target_rotation_jump_deg"]:
                    reasons.append(name)
        return reasons

    def record(
        self,
        started,
        loop_dt_ms,
        timings,
        frame,
        controller_data,
        targets,
        command,
        real_state,
        active,
        mapping_metrics=None,
    ):
        if not self.enabled:
            return
        metrics = {name: round(float(value), 4) for name, value in timings.items()}
        metrics["total_ms"] = round((time.perf_counter() - started) * 1000.0, 4)
        if loop_dt_ms is not None:
            metrics["loop_dt_ms"] = round(float(loop_dt_ms), 4)
        metrics.update(self._sensor_timestamp_gap(controller_data))
        for name, duration in controller_data.get("pose_unchanged_ms", {}).items():
            metrics[f"pose_{name}_unchanged_ms"] = duration
        metrics.update(self._sensor_deltas(frame))
        metrics.update(self._target_deltas(targets))
        if mapping_metrics:
            metrics.update(mapping_metrics)
        jump, self.previous_command = self._array_jump(
            "command_joint_jump_rad", command, self.previous_command
        )
        metrics.update(jump)
        jump, self.previous_real_state = self._array_jump(
            "real_state_jump_rad", real_state, self.previous_real_state
        )
        metrics.update(jump)
        metrics = {name: round(float(value), 4) for name, value in metrics.items()}

        sample = {
            "time": time.time(),
            "active": bool(active),
            "tracking": frame is not None,
            "metrics": metrics,
        }
        self.history.append(sample)
        reasons = self._anomaly_reasons(metrics)
        if reasons:
            now = time.perf_counter()
            if now - self.last_event >= self.event_cooldown_s:
                self.last_event = now
                self._write(
                    {
                        "type": "anomaly",
                        "time": time.time(),
                        "reasons": reasons,
                        "current": sample,
                        "history": list(self.history),
                    }
                )
            if now - self.last_console >= self.console_cooldown_s:
                self.last_console = now
                details = (
                    f"total={metrics.get('total_ms', 0):.1f}ms "
                    f"input={metrics.get('input_ms', 0):.1f}ms "
                    f"ik={metrics.get('ik_ms', 0):.1f}ms "
                    f"redis={max(metrics.get('redis_read_ms', 0), metrics.get('redis_publish_ms', 0)):.1f}ms"
                )
                print(f"[Diag warning] {','.join(reasons)} | {details}")

        self.interval_count += 1
        for name, value in metrics.items():
            self.interval_sums[name] = self.interval_sums.get(name, 0.0) + value
            self.interval_maxima[name] = max(self.interval_maxima.get(name, value), value)
        now = time.perf_counter()
        if now - self.last_summary >= self.summary_interval_s:
            count = max(1, self.interval_count)
            self._write(
                {
                    "type": "summary",
                    "time": time.time(),
                    "frames": self.interval_count,
                    "mean": {
                        name: round(value / count, 4)
                        for name, value in self.interval_sums.items()
                    },
                    "max": {
                        name: round(value, 4)
                        for name, value in self.interval_maxima.items()
                    },
                }
            )
            self.interval_count = 0
            self.interval_sums.clear()
            self.interval_maxima.clear()
            self.last_summary = now

    def _write(self, payload):
        self.file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self):
        if self.file is not None and not self.file.closed:
            self.file.close()

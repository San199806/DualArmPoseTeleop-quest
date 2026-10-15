import hashlib
import json
import pathlib
import time

import mujoco.viewer
import numpy as np

from scipy.spatial.transform import Rotation

from .diagnostics import RuntimeDiagnostics
from .ik_solver import DualArmIKSolver
from .mapper import PoseArmMapper
from .types import ArmTarget, DualArmTargets, Pose
from .skeleton_input import SkeletonInput
from .safety import CommandLimiter, FreshnessWatchdog
from .recording import RecordedInput, RecordingInput
from .configuration import load_configuration, print_summary


class TeleopApplication:
    def __init__(
        self,
        config_path,
        redis_host=None,
        headless=False,
        auto_start=False,
        simulation=None,
        record_path=None,
        replay_path=None,
        robot=None,
        skeleton_url=None,
    ):
        config_path = pathlib.Path(config_path).resolve()
        loaded = load_configuration(config_path,robot=robot)
        self.project_root = loaded.project_root
        self.config = loaded.values
        if redis_host is not None:self.config['runtime']['redis_host'] = redis_host
        self.robot_name = self.config["robot"]
        model_xml = pathlib.Path(self.config["model_xml"])
        if not model_xml.is_absolute():
            model_xml = (self.project_root / model_xml).resolve()
        if not model_xml.exists():
            raise FileNotFoundError(f"Robot model not found: {model_xml}")
        self.model_xml_path = model_xml
        self.simulation = self.config['runtime']['simulation'] if simulation is None else simulation
        self.replay = bool(replay_path)
        if self.replay:self.simulation = True
        self.config['runtime']['simulation'] = self.simulation
        print_summary(loaded)

        if replay_path:
            self.simulation = True
            self.input = RecordedInput(replay_path)
        else:
            url = skeleton_url
            if url is None:
                url = self.config["runtime"].get("skeleton_url") or None
            hold = float(self.config["runtime"].get("calibration_hold_s", 1.0))
            self.input = SkeletonInput(url, key_hold_s=hold+0.25)
        if record_path:
            self.input = RecordingInput(self.input, record_path)
        self.ik = DualArmIKSolver(
            model_xml,
            self.config["frames"],
            self.config["arms"],
            self.config["ik"],
            self.config["initial_qpos"],
            self.config["runtime"].get("command_output", "full_qpos"),
        )
        self.mapper = PoseArmMapper(
            self.config["sensor"],
            self.config["ik"],
            self.config.get("mapping"),
        )
        self.calibration_signature = self._make_calibration_signature()
        self.headless = headless
        self.active = auto_start
        self.last_toggle = False
        self.last_exit = False
        self.calibration_hold_started = None
        self.calibration_triggered = False
        self.calibration_samples = []
        self.calibration_sample_started = None
        self.calibration_last_timestamp = None
        self.calibration_in_progress = False
        self.frame_index = 0
        self._loop_count = 0
        self._loop_window_at = None
        self._loop_window_count = 0
        self.viewer_debug_data = None
        self.last_sensor_timestamp = None
        self.last_sensor_change_time = time.monotonic()
        self.last_complete_frame_time = time.monotonic()
        self.sensor_stale_reported = False
        self.last_safe_command = None
        self.last_applied_qpos = self.ik.initial_qpos.copy()
        self.yaw_joint = self._torso_joint(self.config.get("waist_joint"))
        self.pitch_joint = self._torso_joint(self.config.get("pitch_joint"))
        self.needs_command_seed = True
        self.output_held = True
        self.watchdog = FreshnessWatchdog()
        self.last_controller_data = {}
        self.last_command_velocity = None
        self.command_transition_started = time.monotonic()
        self.last_real_state_payload = None
        self.real_state_warning_printed = False
        self.last_ik_error_warning_time = 0.0
        self.last_collision_warning_time = 0.0
        self.last_glitch_warning_time = 0.0
        self.last_redis_warning_time = 0.0
        self.redis = None
        self.redis_pipeline = None
        runtime = self.config["runtime"]
        self.period = 1.0 / float(runtime["frequency_hz"])
        self.command_dt = self.period
        self._command_clock = None
        self.print_interval = max(1, int(runtime["print_every_n_frames"]))
        self.redis_key = runtime["redis_key"]
        self.controller_key = runtime["controller_key"]
        self.real_joint_state_key = runtime["real_joint_state_key"]
        self.real_joint_state_timestamp_key = runtime.get(
            "real_joint_state_timestamp_key", "t_real_robot_joint_state"
        )
        self.real_joint_state_max_age_ms = float(
            runtime.get("real_joint_state_max_age_ms", 100.0)
        )
        self.command_valid_key = runtime.get(
            "command_valid_key", "teleop_command_valid"
        )
        self.sensor_hold_timeout_ms = float(
            runtime.get("sensor_hold_timeout_ms", 100.0)
        )
        self.sensor_pause_timeout_ms = float(
            runtime.get("sensor_pause_timeout_ms", 500.0)
        )
        self.max_joint_velocity = float(
            runtime.get("max_joint_velocity_rad_s", 1.5)
        )
        self.max_joint_acceleration = float(
            runtime.get("max_joint_acceleration_rad_s2", 6.0)
        )
        self.limiter = CommandLimiter(
            *self.ik.command_bounds(), self.max_joint_velocity, self.max_joint_acceleration
        )
        self.command_ttl_ms = int(runtime.get("command_ttl_ms", 200))
        if self.command_ttl_ms <= 0:
            raise ValueError("command_ttl_ms must be positive")
        self.command_ramp_s = float(runtime.get("command_ramp_s", 0.75))
        self.max_wrist_ik_error_m = float(
            runtime.get("max_wrist_ik_error_m", 0.12)
        )
        self.show_ik_debug = bool(runtime.get("show_ik_debug", True))
        self.ik_debug_target_radius = float(
            runtime.get("ik_debug_target_radius_m", 0.025)
        )
        self.ik_debug_actual_radius = float(
            runtime.get("ik_debug_actual_radius_m", 0.015)
        )
        self.ik_debug_error_radius = float(
            runtime.get("ik_debug_error_radius_m", 0.004)
        )
        calibration_file = pathlib.Path(runtime["calibration_file"])
        self.calibration_path = (
            calibration_file
            if calibration_file.is_absolute()
            else (self.project_root / calibration_file).resolve()
        )
        self.calibration_hold_s = float(runtime.get("calibration_hold_s", 1.0))
        self.require_session_calibration = bool(
            runtime.get("require_session_calibration", True)
        )
        self.session_calibration_duration_s = float(
            runtime.get("session_calibration_duration_s", 1.0)
        )
        self.session_calibration_max_motion_deg = float(
            runtime.get("session_calibration_max_motion_deg", 5.0)
        )
        self.session_calibration_min_samples = int(
            runtime.get("session_calibration_min_samples", 20)
        )
        self.session_calibration_max_position_motion_m = float(
            runtime.get("session_calibration_max_position_motion_m", 0.025)
        )
        self.session_calibration_min_hand_separation_m = float(
            runtime.get("session_calibration_min_hand_separation_m", 0.15)
        )
        self.session_calibration_max_hand_height_diff_m = float(
            runtime.get("session_calibration_max_hand_height_diff_m", 0.12)
        )
        self.session_calibration_max_hand_radial_diff_m = float(
            runtime.get("session_calibration_max_hand_radial_diff_m", 0.12)
        )
        self.diagnostics = RuntimeDiagnostics(
            self.robot_name,
            self.period,
            self.project_root,
            self.config.get("diagnostics"),
        )
        host = redis_host or runtime["redis_host"]
        try:
            import redis
            self.redis = redis.Redis(
                host=host, port=int(runtime["redis_port"]), db=0,
                socket_connect_timeout=0.1, socket_timeout=0.05,
            ) if not replay_path else None
            if replay_path:
                print("[Replay] offline; Redis output disabled")
                return
            self.redis.ping()
            self.redis_pipeline = self.redis.pipeline(transaction=True)
            print(f"[Redis] connected: {host}:{runtime['redis_port']}")
        except Exception as error:
            print(f"[Redis] disabled: {error}")
        self._load_calibration()

    def _make_calibration_signature(self):
        relevant_config = {
            "robot": self.config["robot"],
            "model_sha256": hashlib.sha256(
                self.model_xml_path.read_bytes()
            ).hexdigest(),
            "frames": self.config["frames"],
            "arms": self.config["arms"],
            "sensor": self.config["sensor"],
            "mapping": self.config.get("mapping", {}),
            "initial_qpos": self.config["initial_qpos"],
        }
        canonical = json.dumps(
            relevant_config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _handle_buttons(self, controller_data):
        toggle = bool(controller_data["RightController"]["key_one"])
        exit_pressed = bool(controller_data["LeftController"]["key_one"])
        calibration_pressed = bool(controller_data["LeftController"]["key_two"])
        toggle_edge = toggle and not self.last_toggle
        exit_edge = exit_pressed and not self.last_exit
        self.last_toggle = toggle
        self.last_exit = exit_pressed
        if toggle_edge:
            self.active = not self.active
            self.command_transition_started = time.monotonic()
            self.last_command_velocity = None
            self.output_held = True
            self.needs_command_seed = True
            if not self.active:
                self._reset_calibration_window()
            print(f"[State] {'teleop' if self.active else 'pause'}")

        recalibrate = False
        now = time.monotonic()
        if calibration_pressed:
            if self.calibration_hold_started is None:
                self.calibration_hold_started = now
                if self.active:
                    self.active = False
                    self.last_command_velocity = None
                    print("[State] safe hold for recalibration")
            elif (
                not self.calibration_triggered
                and now - self.calibration_hold_started >= self.calibration_hold_s
            ):
                self.calibration_triggered = True
                recalibrate = True
        else:
            self.calibration_hold_started = None
            self.calibration_triggered = False
        return exit_edge, recalibrate

    def _load_calibration(self):
        if self.require_session_calibration:
            if self.calibration_path.exists():
                print(
                    "[Calibration] previous calibration found; wearable orientation "
                    "will be recalibrated for this session"
                )
            else:
                print("[Calibration] no saved file")
            print(
                "[Calibration] hold the waist and both arms in the configured "
                "L pose, then press A"
            )
            return
        if not self.calibration_path.exists():
            print("[Calibration] no saved file; the first teleop start will calibrate")
            return
        try:
            saved = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            if saved.get("model_xml") != self.config["model_xml"]:
                raise ValueError("robot model does not match")
            if saved.get("profile_signature") != self.calibration_signature:
                raise ValueError("robot profile or initial_qpos changed")
            self.mapper.load_calibration(saved["mapper"], self.ik.calibration_poses())
            print(f"[Calibration] loaded: {self.calibration_path}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.mapper.reset()
            print(f"[Calibration] ignored invalid file: {error}")

    def _save_calibration(self):
        if self.replay:
            return
        saved = {
            "version": 1,
            "model_xml": self.config["model_xml"],
            "profile_signature": self.calibration_signature,
            "mapper": self.mapper.export_calibration(),
        }
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.calibration_path.with_suffix(
            self.calibration_path.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.calibration_path)
        print(f"[Calibration] saved: {self.calibration_path}")

    def _calibrate(self, frame):
        # Mapping reference is always the configured simulation pose.
        current = self.ik.configuration.data.qpos.copy()
        self.ik.reset_to_initial()
        try:
            self.mapper.calibrate(frame, self.ik.calibration_poses())
        finally:
            self.ik.configuration.update(q=current)
        self.output_held = True
        self.needs_command_seed = True
        self._save_calibration()
        for side, calibration in self.mapper.arms.items():
            print(
                f"[Calibration] {side}: upper={calibration.upper_length:.4f} m, "
                f"forearm={calibration.forearm_length:.4f} m"
            )
            if self.mapper.skeleton_segments:
                human_upper = self.mapper.human_upper_lengths[side]
                human_forearm = self.mapper.human_forearm_lengths[side]
                print(
                    f"[Calibration] {side} human: upper={human_upper:.4f} m, "
                    f"forearm={human_forearm:.4f} m, "
                    f"scale=({calibration.upper_length/human_upper:.3f}, "
                    f"{calibration.forearm_length/human_forearm:.3f})"
                )
        human_shoulders = {
            side: frame.head.position
            + frame.waist.rotation.apply(
                self.mapper.arms[side].head_to_shoulder_body
            )
            for side in ("left", "right")
        }
        left_shoulder = human_shoulders["left"]
        right_shoulder = human_shoulders["right"]
        print(
            "[Calibration] virtual shoulders "
            f"L=({left_shoulder[0]:.3f}, {left_shoulder[1]:.3f}, "
            f"{left_shoulder[2]:.3f}) "
            f"R=({right_shoulder[0]:.3f}, {right_shoulder[1]:.3f}, "
            f"{right_shoulder[2]:.3f}) "
            f"height_diff={abs(left_shoulder[2] - right_shoulder[2]):.4f}m"
        )

    def _reset_calibration_window(self):
        self.calibration_samples.clear()
        self.calibration_sample_started = None
        self.calibration_last_timestamp = None

    def _collect_session_calibration(self, frame, sensor_timestamp):
        now = time.monotonic()
        if self.calibration_sample_started is None:
            self.calibration_sample_started = now
            self.calibration_samples = []
            print(
                "[Calibration] sampling session zero; keep the waist and both "
                "arms still in the L pose"
            )
        if (
            sensor_timestamp is not None
            and sensor_timestamp == self.calibration_last_timestamp
        ):
            return False
        if sensor_timestamp is not None:
            self.calibration_last_timestamp = sensor_timestamp
        self.calibration_samples.append(frame)
        if now - self.calibration_sample_started < self.session_calibration_duration_s:
            return False
        if len(self.calibration_samples) < self.session_calibration_min_samples:
            print(
                f"[Calibration] only {len(self.calibration_samples)} fresh frames "
                f"received; need {self.session_calibration_min_samples}; retrying"
            )
            self._reset_calibration_window()
            return False

        motion = self.mapper.calibration_motion_deg(self.calibration_samples)
        if motion > self.session_calibration_max_motion_deg:
            print(
                f"[Calibration] motion too large ({motion:.1f} deg > "
                f"{self.session_calibration_max_motion_deg:.1f} deg); retrying"
            )
            self._reset_calibration_window()
            return False

        position_motion = self.mapper.calibration_position_motion_m(
            self.calibration_samples
        )
        if position_motion > self.session_calibration_max_position_motion_m:
            print(
                f"[Calibration] position motion too large ({position_motion:.3f} m > "
                f"{self.session_calibration_max_position_motion_m:.3f} m); retrying"
            )
            self._reset_calibration_window()
            return False

        averaged_frame = self.mapper.average_frame(self.calibration_samples)
        geometry = self.mapper.calibration_geometry(averaged_frame)
        if geometry["hand_separation_m"] < self.session_calibration_min_hand_separation_m:
            print(
                "[Calibration] hands are too close "
                f"({geometry['hand_separation_m']:.3f} m); retrying"
            )
            self._reset_calibration_window()
            return False
        if geometry["hand_height_diff_m"] > self.session_calibration_max_hand_height_diff_m:
            print(
                "[Calibration] hand height difference is too large "
                f"({geometry['hand_height_diff_m']:.3f} m); retrying"
            )
            self._reset_calibration_window()
            return False
        if geometry["hand_radial_diff_m"] > self.session_calibration_max_hand_radial_diff_m:
            print(
                "[Calibration] left/right hand depth is too asymmetric "
                f"({geometry['hand_radial_diff_m']:.3f} m); retrying"
            )
            self._reset_calibration_window()
            return False
        try:
            self._calibrate(averaged_frame)
        except ValueError as error:
            print(f"[Calibration] rejected: {error}")
            self.active = False
            self.calibration_in_progress = False
            self._reset_calibration_window()
            return False
        self._reset_calibration_window()
        print("[Calibration] session zero ready")
        return True

    def _read_real_arm_joints(self):
        if self.redis is None:
            return None
        try:
            raw = self.redis.get(self.real_joint_state_key)
        except Exception as error:
            self._redis_warning(f"read failed: {error}")
            return None
        if raw is None:
            return None
        try:
            decoded = json.loads(raw)
            embedded_timestamp = None
            if isinstance(decoded, dict):
                embedded_timestamp = decoded.get("timestamp_ms")
                decoded = decoded.get("positions")
            joints = np.asarray(decoded, dtype=np.float64)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"[Redis] invalid {self.real_joint_state_key}: {error}")
            return None
        expected = self.ik.arm_dof_count
        if joints.shape != (expected,) or not np.all(np.isfinite(joints)):
            print(
                f"[Redis] invalid {self.real_joint_state_key}: "
                f"expected {expected} finite values, got shape {joints.shape}"
            )
            return None
        timestamp_ms = embedded_timestamp
        if timestamp_ms is None and self.real_joint_state_timestamp_key:
            try:
                timestamp_raw = self.redis.get(self.real_joint_state_timestamp_key)
            except Exception as error:
                self._redis_warning(f"timestamp read failed: {error}")
                timestamp_raw = None
            if timestamp_raw is not None:
                try:
                    timestamp_ms = float(timestamp_raw)
                except (TypeError, ValueError):
                    timestamp_ms = None
        if timestamp_ms is not None:
            try:
                timestamp_ms = float(timestamp_ms)
            except (ValueError, TypeError):
                return None
            if not np.isfinite(timestamp_ms):
                return None
            age_ms = time.time() * 1000.0 - timestamp_ms
            if age_ms < -1000.0 or age_ms > self.real_joint_state_max_age_ms:
                if not self.real_state_warning_printed:
                    print(f"[Redis] stale real joint state ignored: age={age_ms:.1f} ms")
                    self.real_state_warning_printed = True
                return None
        elif not self.simulation:
            self._redis_warning("feedback has no timestamp; cannot safely seed output")
            return None
        self.real_state_warning_printed = False
        self.last_real_state_payload = raw
        return joints

    def _redis_warning(self, message):
        now = time.monotonic()
        if now - self.last_redis_warning_time >= 1.0:
            print(f"[Redis] {message}")
            self.last_redis_warning_time = now

    def _publish(self, controller_data, qpos=None, command_valid=False):
        if self.redis_pipeline is None:
            return self.simulation
        try:
            self.redis_pipeline.set(self.controller_key, json.dumps(controller_data))
            if qpos is not None:
                self.redis_pipeline.set(self.redis_key, json.dumps(qpos.tolist()))
                self.redis_pipeline.set("t_action", int(time.time() * 1000))
            self.redis_pipeline.set(
                self.command_valid_key, int(bool(command_valid)), px=self.command_ttl_ms
            )
            self.redis_pipeline.execute()
            return True
        except Exception as error:
            self.redis_pipeline.reset()
            self._redis_warning(f"publish failed: {error}")
            self.active = False
            self.output_held = True
            self.needs_command_seed = True
            return False

    def _sensor_stale_ms(self, controller_data):
        now = time.monotonic()
        timestamp = controller_data.get("timestamp")
        age = self.watchdog.age_ms("global", timestamp, now)
        for name, value in controller_data.get("device_timestamps", {}).items():
            age = max(age, self.watchdog.age_ms(name, value, now))
        return age

    def _note_command_dt(self):
        now = time.perf_counter()
        if self._command_clock is not None:
            self.command_dt = float(np.clip(now - self._command_clock, 1e-3, 0.1))
        self._command_clock = now

    def _limit_command(self, desired):
        elapsed = max(0.0, time.monotonic() - self.command_transition_started)
        ramp = min(1.0, elapsed / max(self.command_ramp_s, self.period))
        command, self.pending_velocity = self.limiter.propose(
            desired, self.command_dt, ramp
        )
        return command

    def _step(self):
        self._note_command_dt()
        step_started, loop_dt_ms = self.diagnostics.begin_step()
        self.viewer_debug_data = None
        timings = {}
        stage_started = time.perf_counter()
        frame, controller_data = self.input.read()
        self.last_controller_data = controller_data
        if getattr(self.input, "finished", False):
            return False
        timings["input_ms"] = (time.perf_counter() - stage_started) * 1000.0
        sensor_stale_ms = self._sensor_stale_ms(controller_data)
        timings["sensor_stale_ms"] = sensor_stale_ms
        now = time.monotonic()
        if frame is not None:
            self.last_complete_frame_time = now
        tracking_stale_ms = (now - self.last_complete_frame_time) * 1000.0
        timings["tracking_stale_ms"] = tracking_stale_ms
        exit_requested, recalibrate = self._handle_buttons(controller_data)
        if exit_requested:
            stage_started = time.perf_counter()
            self._publish(controller_data, command_valid=False)
            timings["redis_publish_ms"] = (
                time.perf_counter() - stage_started
            ) * 1000.0
            self.diagnostics.record(
                step_started,
                loop_dt_ms,
                timings,
                frame,
                controller_data,
                None,
                None,
                None,
                self.active,
            )
            return False
        qpos = None
        targets = None
        real_arm_joints = None
        stream_stale_ms = max(sensor_stale_ms, tracking_stale_ms)
        if stream_stale_ms >= self.sensor_pause_timeout_ms and self.active:
            self.active = False
            self.last_command_velocity = None
            print(
                f"[Safety] tracking unavailable for {stream_stale_ms:.0f} ms; teleoperation paused"
            )
        sensor_ready = stream_stale_ms < self.sensor_hold_timeout_ms
        if sensor_ready:
            self.sensor_stale_reported = False
        elif not self.sensor_stale_reported:
            print(
                f"[Safety] sensor data stale for {stream_stale_ms:.0f} ms; command held"
            )
            self.sensor_stale_reported = True
        if frame is not None and not self.simulation and (self.active or recalibrate):
            stage_started = time.perf_counter()
            real_arm_joints = self._read_real_arm_joints()
            timings["redis_read_ms"] = (
                time.perf_counter() - stage_started
            ) * 1000.0
            if real_arm_joints is not None:
                self.ik.inject_real_arm_joints(real_arm_joints)
                if self.needs_command_seed:
                    self.last_applied_qpos = self.ik.configuration.data.qpos.copy()

        if recalibrate:
            stage_started = time.perf_counter()
            if frame is None:
                print("[Calibration] failed: Quest pose is incomplete")
            else:
                self.mapper.reset()
                self._reset_calibration_window()
                self.calibration_in_progress = True
                print("[Calibration] manual session recalibration requested")
            timings["calibration_ms"] = (
                time.perf_counter() - stage_started
            ) * 1000.0

        if (
            frame is not None
            and sensor_ready
            and controller_data.get("timestamp") is not None
            and self.calibration_in_progress
            and not self.active
        ):
            stage_started = time.perf_counter()
            calibration_ready = self._collect_session_calibration(
                frame, controller_data.get("timestamp")
            )
            timings["calibration_ms"] = (
                time.perf_counter() - stage_started
            ) * 1000.0
            if calibration_ready:
                self.calibration_in_progress = False
                print("[Calibration] paused; press A to start teleoperation")
            stage_started = time.perf_counter()
            self._publish(controller_data, command_valid=False)
            timings["redis_publish_ms"] = (
                time.perf_counter() - stage_started
            ) * 1000.0
            self.diagnostics.record(
                step_started,
                loop_dt_ms,
                timings,
                frame,
                controller_data,
                None,
                None,
                real_arm_joints,
                self.active,
            )
            return True

        if frame is not None and self.active and sensor_ready:
            if self.needs_command_seed and not self.simulation and real_arm_joints is None:
                self._redis_warning("waiting for fresh measured joints before starting")
                self._publish(controller_data)
                return True
            if self.output_held or self.needs_command_seed:
                self.limiter.reset(self.ik.command_positions(self.last_applied_qpos))
                self.command_transition_started = time.monotonic()
                self.output_held = False
                self.needs_command_seed = False
            if not self.mapper.calibrated:
                stage_started = time.perf_counter()
                calibration_ready = self._collect_session_calibration(
                    frame, controller_data.get("timestamp")
                )
                timings["calibration_ms"] = (
                    time.perf_counter() - stage_started
                ) * 1000.0
                if not calibration_ready:
                    stage_started = time.perf_counter()
                    self._publish(controller_data, command_valid=False)
                    timings["redis_publish_ms"] = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                    self.diagnostics.record(
                        step_started, loop_dt_ms, timings, frame, controller_data,
                        None, None, real_arm_joints, self.active,
                    )
                    return True
                self.calibration_in_progress = False

            stage_started = time.perf_counter()
            tracked = self.mapper.stabilize_sensor(frame)
            if isinstance(self.mapper.sensor_glitch, str) and self.mapper.sensor_glitch:
                now = time.monotonic()
                if now - self.last_glitch_warning_time >= 1.0:
                    print(f"[Safety] tracking glitch held: {self.mapper.sensor_glitch}")
                    self.last_glitch_warning_time = now
            targets = self.mapper.map(tracked)
            self._set_waist_override(tracked)
            targets = self._rotate_targets_with_waist(targets)
            timings["mapping_ms"] = (
                time.perf_counter() - stage_started
            ) * 1000.0
            stage_started = time.perf_counter()
            qpos = self.ik.solve(targets)
            if qpos is not None and self.mapper.at_calibration_pose:
                qpos = self._qpos_toward_initial(self.last_applied_qpos)
                self.ik.configuration.update(q=qpos)
                self.ik.collision_blocked = None
            timings["ik_ms"] = (time.perf_counter() - stage_started) * 1000.0
            self.viewer_debug_data = self._ik_debug_data(targets)
            wrist_errors = [
                point["error_m"]
                for point in self.viewer_debug_data
                if point["joint"] == "wrist"
            ]
            timings["ik_wrist_error_m"] = max(wrist_errors, default=0.0)
            held_qpos = self.last_applied_qpos
            self.ik.configuration.update(q=held_qpos)
            held_wrist_error = max(
                (point["error_m"] for point in self._ik_debug_data(targets) if point["joint"] == "wrist"),
                default=0.0,
            )
            if qpos is not None:
                self.ik.configuration.update(q=qpos)
            timings["held_wrist_error_m"] = held_wrist_error
            if self.frame_index % self.print_interval == 0:
                left = targets.left.wrist.position
                right = targets.right.wrist.position
                print(
                    f"[Wrist target] L=({left[0]:.3f}, {left[1]:.3f}, {left[2]:.3f}) "
                    f"R=({right[0]:.3f}, {right[1]:.3f}, {right[2]:.3f})"
                )
            self.frame_index += 1
        command = None if qpos is None else self.ik.command_positions(qpos)
        collision_message = getattr(self.ik,"collision_blocked",None)
        if collision_message:
            command = None
        if command is not None:
            if (
                not self.mapper.at_calibration_pose
                and self._wrist_solution_rejected(
                    timings.get("ik_wrist_error_m", 0.0),
                    timings.get("held_wrist_error_m", 0.0),
                )
            ):
                now = time.monotonic()
                if now - self.last_ik_error_warning_time >= 1.0:
                    print(
                        "[Safety] IK wrist error too large "
                        f"({timings['ik_wrist_error_m']:.3f} m); command held"
                    )
                    self.last_ik_error_warning_time = now
                command = None
            else:
                command = self._limit_command(command)
        guard = getattr(self.ik,"collision",None)
        if command is not None and guard is not None and guard.enabled:
            checked_at = time.perf_counter()
            proposed = self.ik.expand_command(command,self.last_applied_qpos)
            report = guard.check_path(self.last_applied_qpos,proposed)
            # Also check the measured start, not only the previous command.
            if report.safe and real_arm_joints is not None:
                measured = self.last_applied_qpos.copy()
                indices = self.ik.arm_qpos_indices['left']+self.ik.arm_qpos_indices['right']
                measured[indices] = real_arm_joints
                report = guard.check_path(measured,proposed)
            timings['collision_check_ms'] = (time.perf_counter()-checked_at)*1000
            timings['collision_distance_m'] = report.distance_m
            if not report.allowed:
                collision_message = report.message
                command = None
        if collision_message:
            timings['collision_blocked'] = 1.0
            now = time.monotonic()
            if now-getattr(self,'last_collision_warning_time',0.0) >= 1:
                print('[Collision] command held: '+collision_message)
                self.last_collision_warning_time = now
        stage_started = time.perf_counter()
        command_valid = command is not None and self.active and sensor_ready
        published = self._publish(controller_data, command, command_valid=command_valid)
        if command_valid and published:
            self.limiter.commit(command, self.pending_velocity)
            self.last_safe_command = command.copy()
            self.last_applied_qpos = self.ik.expand_command(command, self.last_applied_qpos)
            for index, value in self.ik.qpos_overrides.items():
                self.last_applied_qpos[index] = value
        elif not (self.active and sensor_ready):
            self.output_held = True
            self.needs_command_seed = True
        self.ik.configuration.update(q=self.last_applied_qpos)
        if targets is not None:
            self.viewer_debug_data = self._ik_debug_data(targets)
            timings["command_wrist_error_m"] = max(
                p["error_m"] for p in self.viewer_debug_data if p["joint"] == "wrist"
            )
            timings["command_emergency_brake"] = float(self.limiter.emergency_brake)
        timings["redis_publish_ms"] = (
            time.perf_counter() - stage_started
        ) * 1000.0
        self.diagnostics.record(
            step_started,
            loop_dt_ms,
            timings,
            frame,
            controller_data,
            targets,
            command,
            real_arm_joints,
            self.active,
            self.mapper.last_debug_metrics if targets is not None else None,
        )
        return True

    def _torso_joint(self, name):
        if not name:
            return None
        model = self.ik.model
        joint_id = int(model.joint(name).id)
        data = self.ik.configuration.data
        body_id = int(model.jnt_bodyid[joint_id])
        return {
            "index": int(model.jnt_qposadr[joint_id]),
            "range": np.asarray(model.jnt_range[joint_id], dtype=float),
            "anchor": data.xpos[body_id].copy(),
            "axis": data.xaxis[joint_id].copy(),
        }

    def _command_torso_joint(self, joint, measurement, calibration):
        if joint is None or calibration is None:
            return
        index = joint["index"]
        self.ik.qpos_overrides[index] = PoseArmMapper.torso_joint_target(
            measurement,
            calibration,
            float(self.ik.initial_qpos[index]),
            float(joint["range"][0]),
            float(joint["range"][1]),
            float(self.last_applied_qpos[index]),
            self.max_joint_velocity * self.command_dt,
        )

    def _torso_delta(self, joint):
        if joint is None:
            return 0.0
        index = joint["index"]
        return (
            self.ik.qpos_overrides.get(index, float(self.ik.initial_qpos[index]))
            - float(self.ik.initial_qpos[index])
        )

    def _set_waist_override(self, frame):
        if self.yaw_joint is not None:
            self._command_torso_joint(
                self.yaw_joint,
                PoseArmMapper.yaw_of(frame.waist.rotation),
                self.mapper.calibration_yaw,
            )
        if self.pitch_joint is not None:
            pitch = PoseArmMapper._frame_torso_pitch(frame)
            if pitch is not None:
                self._command_torso_joint(
                    self.pitch_joint, pitch, self.mapper.calibration_pitch
                )

    def _rotate_targets_with_waist(self, targets):
        pitch = self._torso_delta(self.pitch_joint)
        yaw = self._torso_delta(self.yaw_joint)
        if abs(pitch) < 1e-8 and abs(yaw) < 1e-8:
            return targets
        pitch_turn = Rotation.from_rotvec(self.pitch_joint["axis"] * pitch) if abs(pitch) >= 1e-8 else None
        yaw_turn = Rotation.from_euler("z", yaw) if abs(yaw) >= 1e-8 else None

        def spin(pose):
            position = np.asarray(pose.position, dtype=float)
            rotation = pose.rotation
            if pitch_turn is not None:
                anchor = self.pitch_joint["anchor"]
                position = anchor + pitch_turn.apply(position - anchor)
                rotation = pitch_turn * rotation
            if yaw_turn is not None:
                position = yaw_turn.apply(position)
                rotation = yaw_turn * rotation
            return Pose(position, rotation)

        return DualArmTargets(
            ArmTarget(spin(targets.left.elbow), spin(targets.left.wrist)),
            ArmTarget(spin(targets.right.elbow), spin(targets.right.wrist)),
        )

    def _qpos_toward_initial(self, start):
        """Move toward the calibrated L pose along a collision-free joint path."""
        goal = self.ik.initial_qpos.copy()
        for index, value in self.ik.qpos_overrides.items():
            goal[index] = value
        start = np.asarray(start, dtype=np.float64).copy()
        guard = getattr(self.ik, "collision", None)
        if guard is None or not guard.enabled:
            return goal
        if guard.check_path(start, goal).allowed:
            return goal
        best = start
        low, high = 0.0, 1.0
        for _ in range(6):
            mid = 0.5 * (low + high)
            candidate = start + mid * (goal - start)
            if guard.check_path(start, candidate).allowed:
                best = candidate
                low = mid
            else:
                high = mid
        return best

    def _wrist_solution_rejected(self, solved_error, held_error):
        """Hold only a solution that is still far and clearly worse than the pose on screen."""
        return (
            solved_error > self.max_wrist_ik_error_m
            and solved_error > held_error + 0.02
        )

    def _ik_debug_data(self, targets):
        points = []
        for side in ("left", "right"):
            arm_target = getattr(targets, side)
            for joint_name in ("shoulder", "elbow", "wrist"):
                actual = self.ik.body_pose(
                    self.config["frames"][f"{side}_{joint_name}"]
                ).position
                if joint_name == "shoulder":
                    target = actual.copy()
                else:
                    target = getattr(arm_target, joint_name).position.copy()
                points.append(
                    {
                        "side": side,
                        "joint": joint_name,
                        "target": target,
                        "actual": actual.copy(),
                        "error_m": float(np.linalg.norm(target - actual)),
                    }
                )
        return points

    @staticmethod
    def _viewer_sphere(scene, position, radius, color):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, radius, dtype=np.float64),
            np.asarray(position, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1

    @staticmethod
    def _viewer_connector(scene, start, end, radius, color):
        if scene.ngeom >= scene.maxgeom:
            return
        if np.linalg.norm(np.asarray(end) - np.asarray(start)) < 1e-6:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        geom.rgba[:] = np.asarray(color, dtype=np.float32)
        scene.ngeom += 1

    def _update_viewer_debug(self, viewer):
        if not self.show_ik_debug or not hasattr(viewer, "user_scn"):
            return
        with viewer.lock():
            scene = viewer.user_scn
            scene.ngeom = 0
            if self.viewer_debug_data is None:
                return
            side_colors = {
                "left": np.array([0.0, 0.85, 1.0, 0.85]),
                "right": np.array([1.0, 0.55, 0.0, 0.85]),
            }
            actual_color = np.array([1.0, 1.0, 1.0, 0.95])
            error_color = np.array([1.0, 0.05, 0.05, 0.80])
            shoulder_color = np.array([0.65, 0.65, 0.65, 0.80])
            for point in self.viewer_debug_data:
                target_color = (
                    shoulder_color
                    if point["joint"] == "shoulder"
                    else side_colors[point["side"]]
                )
                self._viewer_sphere(
                    scene,
                    point["target"],
                    self.ik_debug_target_radius,
                    target_color,
                )
                self._viewer_sphere(
                    scene,
                    point["actual"],
                    self.ik_debug_actual_radius,
                    actual_color,
                )
                self._viewer_connector(
                    scene,
                    point["actual"],
                    point["target"],
                    self.ik_debug_error_radius,
                    error_color,
                )

    def _report_loop_rate(self):
        now = time.perf_counter()
        self._loop_count += 1
        started = self._loop_window_at
        if started is None:
            self._loop_window_at = now
            self._loop_window_count = self._loop_count
            return
        elapsed = now - started
        if elapsed < 1.0:
            return
        rate_hz = (self._loop_count - self._loop_window_count) / elapsed
        print(f"[Loop] solve {rate_hz:.1f} Hz")
        self._loop_window_at = now
        self._loop_window_count = self._loop_count

    def _on_key(self, keycode):
        target = self.input
        if hasattr(target, "press_key"):
            target.press_key(keycode)

    def run(self):
        print(f"[Robot] {self.robot_name}, nq={self.ik.model.nq}")
        print(f"[Mapping] mode={self.mapper.mode}")
        if self.simulation:
            print("[Mode] simulation: real_robot_joint_state is ignored")
        if getattr(self, "replay", False):
            print("[Controls] A: teleop/pause, hold Y: recalibrate, X: exit")
        else:
            print("[Controls] right A: teleop/pause, hold left Y: recalibrate, left X: exit")
        if self.require_session_calibration:
            print(
                "[Calibration] every process start requires a stationary "
                "session-zero capture"
            )
        if self.show_ik_debug and not self.headless:
            print(
                "[Viewer] IK debug: left target=cyan, right target=orange, "
                "actual=white, position error=red line"
            )
        try:
            if self.headless:
                while True:
                    started = time.perf_counter()
                    if not self._step():
                        break
                    time.sleep(
                        max(0.0, self.period - (time.perf_counter() - started))
                    )
                    self._report_loop_rate()
                return

            with mujoco.viewer.launch_passive(
                self.ik.model,
                self.ik.configuration.data,
                key_callback=self._on_key,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
                while viewer.is_running():
                    started = time.perf_counter()
                    if not self._step():
                        break
                    self._update_viewer_debug(viewer)
                    viewer.sync()
                    time.sleep(
                        max(0.0, self.period - (time.perf_counter() - started))
                    )
                    self._report_loop_rate()
        finally:
            try:
                self._publish(self.last_controller_data, command_valid=False)
            finally:
                self.diagnostics.close()
                if hasattr(self.input, "close"):
                    self.input.close()

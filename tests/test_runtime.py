import pathlib
import sys
import tempfile
import time
import unittest
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/"src"))
from dual_arm_pose_teleop.app import TeleopApplication
from dual_arm_pose_teleop.recording import RecordingInput, RecordedInput
from dual_arm_pose_teleop.configuration import load_configuration


class RuntimeTests(unittest.TestCase):
    def test_step_displays_sent_command_and_rolls_back_on_publish_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = pathlib.Path(directory)/"fake.xml"
            model_path.write_text("<mujoco/>")
            config = load_configuration(pathlib.Path(__file__).resolve().parents[1]/"robots"/"baxter.json").values
            config["model_xml"] = str(model_path)
            config["diagnostics"]["enabled"] = False
            config_path = pathlib.Path(directory)/"profile.json"
            config_path.write_text(json.dumps(config))
            ik = Mock()
            ik.collision_blocked = None
            ik.collision.enabled = False
            ik.initial_qpos = np.zeros(1)
            ik.configuration.data.qpos = np.zeros(1)
            ik.configuration.update.side_effect = lambda q: setattr(ik.configuration.data, "qpos", q.copy())
            ik.command_bounds.return_value = (np.array([-1.]), np.array([1.]))
            ik.command_positions.side_effect = lambda q: q.copy()
            ik.expand_command.side_effect = lambda q, reference: q.copy()
            def solve(targets):
                ik.configuration.update(q=np.ones(1))
                return np.ones(1)
            ik.solve.side_effect = solve
            mapper = Mock()
            mapper.calibrated = True
            mapper.last_debug_metrics = {}
            mapper.map.return_value = SimpleNamespace(
                left=SimpleNamespace(wrist=SimpleNamespace(position=np.zeros(3))),
                right=SimpleNamespace(wrist=SimpleNamespace(position=np.zeros(3))))
            source = Mock()
            source.finished = False
            source.read.side_effect = lambda: (object(), {
                "timestamp": time.time_ns(),
                "LeftController": {"key_one":False,"key_two":False},
                "RightController": {"key_one":False,"key_two":False}})
            redis = Mock()
            with patch("dual_arm_pose_teleop.app.XRobotInput", return_value=source), \
                 patch("dual_arm_pose_teleop.app.DualArmIKSolver", return_value=ik), \
                 patch("dual_arm_pose_teleop.app.PoseArmMapper", return_value=mapper), \
                 patch("redis.Redis", return_value=redis):
                app = TeleopApplication(config_path, simulation=True, auto_start=True)
            app._ik_debug_data = lambda targets: [{"joint":"wrist","error_m":0.0}]
            app._step()
            sent = app.last_applied_qpos.copy()
            self.assertLess(sent[0], .01)
            np.testing.assert_array_equal(ik.configuration.data.qpos, sent)
            redis.pipeline.assert_called_once_with(transaction=True)
            redis.pipeline.return_value.execute.side_effect = TimeoutError("outage")
            app._step()
            np.testing.assert_array_equal(ik.configuration.data.qpos, sent)
            np.testing.assert_array_equal(app.limiter.q, sent)
            self.assertFalse(app.active)
            # Collision rejection must not publish/commit the candidate.
            app.active = True
            redis.pipeline.return_value.execute.side_effect = None
            redis.pipeline.return_value.set.reset_mock()
            ik.collision.enabled = True
            ik.collision.check_path.return_value = SimpleNamespace(
                safe=False,distance_m=-.01,message='test colliding pair')
            app._step()
            self.assertFalse(app.active)
            np.testing.assert_array_equal(app.limiter.q,sent)
            np.testing.assert_array_equal(app.last_applied_qpos,sent)
            self.assertFalse(any(c.args[0] == 't_action' for c in redis.pipeline.return_value.set.call_args_list))
            redis.pipeline.return_value.set.assert_any_call(app.command_valid_key,0,px=app.command_ttl_ms)

    def app(self):
        app = TeleopApplication.__new__(TeleopApplication)
        app.redis_pipeline = Mock()
        app.controller_key = "controller_data"
        app.redis_key = "action"
        app.command_valid_key = "valid"
        app.command_ttl_ms = 200
        app.last_redis_warning_time = 0
        app.simulation = False
        app.active = True
        app.output_held = False
        return app

    def test_pause_does_not_refresh_action_timestamp(self):
        app = self.app()
        self.assertTrue(app._publish({}))
        calls = app.redis_pipeline.set.call_args_list
        self.assertFalse(any(call.args[0] == "t_action" for call in calls))
        app.redis_pipeline.set.assert_any_call("valid", 0, px=200)

    def test_valid_command_has_expiry(self):
        app = self.app()
        self.assertTrue(app._publish({}, np.zeros(2), True))
        app.redis_pipeline.set.assert_any_call("valid", 1, px=200)

    def test_network_failure_pauses(self):
        app = self.app()
        app.redis_pipeline.execute.side_effect = TimeoutError("test outage")
        self.assertFalse(app._publish({}, np.zeros(2), True))
        self.assertFalse(app.active)
        self.assertTrue(app.needs_command_seed)

    def test_window_or_keyboard_exit_invalidates_output(self):
        app = self.app()
        app.robot_name = "test"
        app.ik = Mock()
        app.mapper = Mock()
        app.require_session_calibration = False
        app.show_ik_debug = False
        app.headless = True
        app.period = .01
        app.last_controller_data = {}
        app.diagnostics = Mock()
        app.input = Mock()
        app._step = Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            app.run()
        app.redis_pipeline.set.assert_any_call("valid", 0, px=200)
        app.input.close.assert_called_once()

    def test_record_replay_preserves_dropouts_and_buttons(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)/"test.jsonl"
            source = Mock()
            source.read.return_value = (None, {"timestamp": 42, "button": True})
            recorder = RecordingInput(source, path)
            recorder.read()
            recorder.close()
            replay = RecordedInput(path)
            frame, data = replay.read()
            self.assertIsNone(frame)
            self.assertEqual(data, {"timestamp": 42, "button": True})
            replay.read()
            self.assertTrue(replay.finished)
            replay.close()

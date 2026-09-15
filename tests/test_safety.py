import pathlib
import sys
import unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/"src"))
from dual_arm_pose_teleop.safety import CommandLimiter, FreshnessWatchdog


class SafetyTests(unittest.TestCase):
    def limiter(self):
        return CommandLimiter([-1.0], [1.0], 1.5, 6.0)

    def test_reported_overshoot_regression(self):
        limiter = self.limiter()
        limiter.reset([0.999])
        limiter.v[:] = 1.5
        q, v = limiter.propose([1.0], .01)
        self.assertLessEqual(q[0], 1.0)
        self.assertTrue(limiter.emergency_brake)

    def test_stationary_target_brakes_with_bounded_acceleration(self):
        limiter = self.limiter()
        limiter.reset([0.0])
        for _ in range(500):
            q, v = limiter.propose([.8], .01)
            self.assertLessEqual(q[0], .8)
            self.assertLessEqual(abs(v[0]), 1.5+1e-9)
            self.assertLessEqual(abs(v[0]-limiter.v[0]), .06+1e-9)
            limiter.commit(q, v)
        self.assertAlmostEqual(limiter.q[0], .8, places=6)

    def test_proposal_does_not_advance_until_sent(self):
        limiter = self.limiter()
        limiter.reset([0.0])
        limiter.propose([.8], .01)
        np.testing.assert_array_equal(limiter.q, [0.0])
        np.testing.assert_array_equal(limiter.v, [0.0])

    def test_hold_reset_and_reverse(self):
        limiter = self.limiter()
        limiter.reset([.4])
        limiter.v[:] = 1
        q, v = limiter.propose([-.5], .01)
        self.assertLessEqual(q[0], .4)
        limiter.reset(q)
        q2, v2 = limiter.propose([-.5], .01, .1)
        self.assertLessEqual(abs(v2[0]), .06+1e-9)

    def test_limits_and_nonfinite(self):
        limiter = self.limiter()
        limiter.reset([0.0])
        for _ in range(500):
            q, v = limiter.propose([100.0], .01)
            self.assertLessEqual(q[0], 1.0)
            limiter.commit(q, v)
        with self.assertRaises(ValueError):
            limiter.propose([np.nan], .01)

    def test_group_freeze_while_global_timestamp_advances(self):
        watchdog = FreshnessWatchdog()
        watchdog.age_ms("global", 10, 0)
        watchdog.age_ms("motion", 10, 0)
        self.assertEqual(watchdog.age_ms("global", 20, .2), 0)
        self.assertAlmostEqual(watchdog.age_ms("motion", 10, .2), 200)

    def test_missing_timestamp_is_not_fresh(self):
        watchdog = FreshnessWatchdog()
        watchdog.age_ms("global", 10, 0)
        self.assertAlmostEqual(watchdog.age_ms("global", None, .6), 600)

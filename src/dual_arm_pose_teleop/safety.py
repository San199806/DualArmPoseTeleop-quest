"""Command trajectory and freshness checks, independent of the device SDK."""
import numpy as np


class CommandLimiter:
    def __init__(self, lower, upper, velocity, acceleration):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.vmax = float(velocity)
        self.amax = float(acceleration)
        if self.vmax <= 0 or self.amax <= 0:
            raise ValueError("Command speed and acceleration must be positive")
        self.q = None
        self.v = None
        self.emergency_brake = False

    def reset(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != self.lower.shape or not np.all(np.isfinite(q)):
            raise ValueError("Invalid command seed")
        self.q = np.clip(q, self.lower, self.upper)
        self.v = np.zeros_like(q)

    def propose(self, desired, dt, ramp=1.0):
        desired = np.asarray(desired, dtype=float)
        if desired.shape != self.q.shape or not np.all(np.isfinite(desired)):
            raise ValueError("Invalid command target")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid trajectory timestep")
        target = np.clip(desired, self.lower, self.upper)
        distance = target - self.q
        # Discrete stopping-distance allowance, including the next sample.
        brake_speed = np.sqrt((self.amax * dt)**2 + 2*self.amax*np.abs(distance)) - self.amax*dt
        wanted = np.sign(distance) * np.minimum(brake_speed, self.vmax*np.clip(ramp, 0, 1))
        velocity = np.clip(wanted, self.v-self.amax*dt, self.v+self.amax*dt)
        candidate = self.q + velocity*dt
        # An abruptly moved target can make acceleration and no-overshoot
        # incompatible. Position/limit containment takes priority in that case.
        bounded = np.clip(candidate, np.minimum(self.q, target), np.maximum(self.q, target))
        bounded = np.clip(bounded, self.lower, self.upper)
        self.emergency_brake = bool(np.any(np.abs(candidate-bounded) > 1e-10))
        return bounded, (bounded-self.q)/dt

    def commit(self, q, velocity):
        self.q = q.copy()
        self.v = velocity.copy()


class FreshnessWatchdog:
    def __init__(self):
        self.seen = {}

    def age_ms(self, key, timestamp, now):
        previous, changed = self.seen.get(key, (None, now))
        if timestamp is not None and (previous is None or timestamp > previous):
            changed = now
        self.seen[key] = (timestamp if timestamp is not None else previous, changed)
        return max(0.0, (now-changed)*1000)

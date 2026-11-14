"""Discrete-model velocity control and shadow cargo identification."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _vector(value) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(2)
    if not np.isfinite(result).all():
        raise ValueError("velocity or command must be finite")
    return result


def unit_disk(value) -> np.ndarray:
    vector = _vector(value)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0 else vector.copy()


def world_to_body(u_world, heading: float) -> np.ndarray:
    x, y = _vector(u_world)
    if not math.isfinite(heading):
        raise ValueError("heading must be finite")
    c, s = math.cos(heading), math.sin(heading)
    return np.array((c * x + s * y, -s * x + c * y))


def body_to_world(u_body, heading: float) -> np.ndarray:
    x, y = _vector(u_body)
    if not math.isfinite(heading):
        raise ValueError("heading must be finite")
    c, s = math.cos(heading), math.sin(heading)
    return np.array((c * x - s * y, s * x + c * y))


@dataclass(frozen=True)
class PlanarModel:
    decay: float
    thrust_gain: float

    def __post_init__(self):
        if not (0.0 < self.decay < 1.0 and self.thrust_gain > 0.0):
            raise ValueError("invalid dynamics parameters")


EMPTY_MODEL = PlanarModel(0.95, 0.5729312289729012)


class PolePlacementVelocityController:
    def __init__(self, model: PlanarModel = EMPTY_MODEL,
                 closed_loop_pole: float = 0.75):
        if not 0.0 < closed_loop_pole < 1.0:
            raise ValueError("invalid pole")
        self.model = model
        self.pole = float(closed_loop_pole)

    def compute(self, desired_velocity, estimated_velocity) -> np.ndarray:
        desired = _vector(desired_velocity)
        measured = _vector(estimated_velocity)
        a, b = self.model.decay, self.model.thrust_gain
        feedforward = ((1.0 - a) / b) * desired
        feedback = ((a - self.pole) / b) * (desired - measured)
        return unit_disk(feedforward + feedback)


def reference_velocity(position, waypoint, *, max_speed: float,
                       braking_deceleration: float,
                       target_tolerance: float = 5.0,
                       clearance_speed_cap: float | None = None) -> np.ndarray:
    if (not math.isfinite(max_speed) or max_speed < 0
            or not math.isfinite(braking_deceleration)
            or braking_deceleration < 0
            or not math.isfinite(target_tolerance)
            or target_tolerance < 0):
        raise ValueError("invalid speed or braking parameter")
    pos, goal = _vector(position), _vector(waypoint)
    delta = goal - pos
    distance = float(np.linalg.norm(delta))
    if distance <= target_tolerance:
        return np.zeros(2)
    speed = min(max_speed, math.sqrt(2.0 * braking_deceleration
                                     * max(0.0, distance - target_tolerance)))
    if clearance_speed_cap is not None:
        if not math.isfinite(clearance_speed_cap):
            raise ValueError("clearance speed cap must be finite")
        speed = min(speed, max(0.0, clearance_speed_cap))
    return speed * delta / distance


class SharedAxisRLS:
    """Two-axis shared linear fit; never authorizes control by itself."""

    def __init__(self, initial: PlanarModel = PlanarModel(0.95, 0.19),
                 forgetting: float = 0.998):
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("invalid forgetting factor")
        self.theta = np.array([initial.decay, initial.thrust_gain], dtype=float)
        self.P = np.eye(2) * 30.0
        self.forgetting = float(forgetting)
        self.updates = 0
        self.error_squares = 0.0

    def observe(self, previous_velocity, applied_world_thrust, new_velocity,
                *, valid_free_motion: bool) -> bool:
        if not valid_free_motion:
            return False
        try:
            v0 = _vector(previous_velocity)
            thrust = _vector(applied_world_thrust)
            v1 = _vector(new_velocity)
        except (ValueError, TypeError):
            return False
        if max(float(np.linalg.norm(v0)), float(np.linalg.norm(thrust))) < 0.15:
            return False
        residual = v1 - self.theta[0] * v0 - self.theta[1] * thrust
        if self.updates > 20 and float(np.linalg.norm(residual)) > 2.0:
            return False
        accepted = False
        for axis in range(2):
            phi = np.array([v0[axis], thrust[axis]])
            denom = self.forgetting + float(phi @ self.P @ phi)
            if not math.isfinite(denom) or denom <= 0.0:
                continue
            gain = self.P @ phi / denom
            error = float(v1[axis] - phi @ self.theta)
            candidate = self.theta + gain * error
            if 0.65 <= candidate[0] <= 0.999 and 0.025 <= candidate[1] <= 0.80:
                self.theta = candidate
                self.P = (self.P - np.outer(gain, phi) @ self.P) / self.forgetting
                self.error_squares += error * error
                accepted = True
        if accepted:
            self.updates += 1
        return accepted

    @property
    def rmse(self) -> float:
        return math.sqrt(self.error_squares / max(1, 2 * self.updates))

    def validated_model(self, *, min_updates: int = 100,
                        max_rmse: float = 0.3) -> PlanarModel | None:
        if self.updates < min_updates or self.rmse > max_rmse:
            return None
        return PlanarModel(float(self.theta[0]), float(self.theta[1]))


class ShadowCargoIdentifier:
    """Fit legal, consecutive free-motion samples without changing actions.

    ``measured_velocity`` is the odometer displacement over the *last*
    control interval.  ``applied_world_thrust`` is the final command about to
    run in the *next* interval.  Thus the stored previous thrust is aligned
    with the next call's measured displacement.  The fitted coefficients are
    for this observable frame-average quantity, not asserted endpoint truth.
    """

    def __init__(self):
        self.rls = SharedAxisRLS()
        self.previous = None
        self.eligible_pairs = 0
        self.rejected_pairs = 0
        self.holdout_pairs = 0
        self.holdout_error_squares = 0.0

    def advance(self, step: int, measured_velocity, applied_world_thrust,
                *, eligible: bool) -> bool:
        try:
            velocity = _vector(measured_velocity)
            thrust = unit_disk(applied_world_thrust)
        except (ValueError, TypeError):
            self.previous = None
            self.rejected_pairs += 1
            return False
        accepted = False
        if self.previous is not None:
            previous_step, previous_velocity, previous_thrust, previous_ok = self.previous
            pair_ok = (step == previous_step + 1 and previous_ok and eligible)
            if pair_ok:
                self.eligible_pairs += 1
                if self.eligible_pairs % 5 == 0:
                    prediction = (self.rls.theta[0] * previous_velocity
                                  + self.rls.theta[1] * previous_thrust)
                    self.holdout_error_squares += float(np.sum((velocity - prediction) ** 2))
                    self.holdout_pairs += 1
                else:
                    accepted = self.rls.observe(
                        previous_velocity, previous_thrust, velocity,
                        valid_free_motion=True,
                    )
            else:
                self.rejected_pairs += 1
        self.previous = (int(step), velocity, thrust, bool(eligible))
        return accepted

    def diagnostics(self) -> dict:
        return {
            "eligible_pairs": self.eligible_pairs,
            "rejected_pairs": self.rejected_pairs,
            "updates": self.rls.updates,
            "decay_candidate": float(self.rls.theta[0]),
            "thrust_gain_candidate": float(self.rls.theta[1]),
            "training_rmse": self.rls.rmse,
            "holdout_pairs": self.holdout_pairs,
            "holdout_rmse": math.sqrt(self.holdout_error_squares
                                      / max(1, 2 * self.holdout_pairs)),
            "control_authorized": False,
        }

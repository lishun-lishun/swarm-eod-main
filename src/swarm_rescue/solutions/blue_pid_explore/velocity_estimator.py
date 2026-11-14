"""Legal model/odometry/GPS velocity and position estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .dynamics import (
    INPUT_GAIN,
    POSITION_INPUT_GAIN,
    POSITION_VELOCITY_GAIN,
    VELOCITY_DECAY,
)


def _valid_vector(value, size: int) -> bool:
    if value is None:
        return False
    array = np.asarray(value, dtype=float)
    return array.shape == (size,) and bool(np.all(np.isfinite(array)))


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _blend_angle(predicted: float, observed: float, gain: float) -> float:
    return _wrap(predicted + gain * _wrap(observed - predicted))


@dataclass(frozen=True)
class StateEstimate:
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    used_odometry: bool
    used_gps: bool
    used_compass: bool


class LegalVelocityEstimator:
    """Endpoint velocity observer using only competition-legal measurements.

    ``update`` is called at the start of a control frame.  Its odometry sample
    describes the displacement produced by ``previous_command_world`` during
    the frame that has just ended.
    """

    def __init__(
        self,
        *,
        odometry_velocity_gain: float = 0.45,
        gps_position_gain: float = 0.002,
        odometry_position_gain: float = 0.0,
        compass_gain: float = 0.08,
    ):
        self.odometry_velocity_gain = float(odometry_velocity_gain)
        self.gps_position_gain = float(gps_position_gain)
        self.odometry_position_gain = float(odometry_position_gain)
        self.compass_gain = float(compass_gain)
        self.reset()

    def reset(self) -> None:
        self._initialized = False
        self._position = np.zeros(2)
        self._velocity = np.zeros(2)
        self._heading = 0.0
        self._previous_command = np.zeros(2)
        self._previous_rotation_command = 0.0
        # Position dead reckoning keeps a model-only velocity so zero-mean
        # odometry velocity corrections do not integrate into position drift.
        self._dead_reckoning_velocity = np.zeros(2)

    @property
    def estimate(self) -> StateEstimate:
        return StateEstimate(self._position.copy(), self._velocity.copy(), self._heading, False, False, False)

    def set_previous_command(self, command_world: Sequence[float], rotation: float = 0.0) -> None:
        command = np.asarray(command_world, dtype=float)
        if command.shape != (2,) or not np.all(np.isfinite(command)):
            raise ValueError("previous command must be a finite 2-vector")
        self._previous_command = command.copy()
        self._previous_rotation_command = float(rotation)

    def update(
        self,
        *,
        odometry: Optional[Sequence[float]],
        compass: Optional[float],
        gps_position: Optional[Sequence[float]],
    ) -> StateEstimate:
        odom_ok = _valid_vector(odometry, 3)
        gps_ok = _valid_vector(gps_position, 2)
        compass_ok = compass is not None and bool(np.isfinite(float(compass)))

        if not self._initialized:
            if gps_ok:
                self._position = np.asarray(gps_position, dtype=float).copy()
            if compass_ok:
                self._heading = _wrap(float(compass))
            self._velocity[:] = 0.0
            self._initialized = True
            return StateEstimate(self._position.copy(), self._velocity.copy(), self._heading, False, gps_ok, compass_ok)

        previous_velocity = self._velocity.copy()
        dead_reckoning_velocity = self._dead_reckoning_velocity.copy()
        predicted_velocity = VELOCITY_DECAY * previous_velocity + INPUT_GAIN * self._previous_command
        predicted_position = (
            self._position
            + POSITION_VELOCITY_GAIN * dead_reckoning_velocity
            + POSITION_INPUT_GAIN * self._previous_command
        )
        self._dead_reckoning_velocity = (
            VELOCITY_DECAY * dead_reckoning_velocity + INPUT_GAIN * self._previous_command
        )

        theta_measured = float(np.asarray(odometry, dtype=float)[2]) if odom_ok else 0.0
        rotating = abs(self._previous_rotation_command) > 1e-9
        theta = theta_measured if rotating else 0.0
        predicted_heading = _wrap(self._heading + theta)
        if compass_ok and rotating:
            self._heading = _blend_angle(predicted_heading, float(compass), self.compass_gain)
        else:
            self._heading = predicted_heading

        if odom_ok:
            distance, alpha, _ = np.asarray(odometry, dtype=float)
            # alpha is expressed from the previous body frame.  Undo the
            # odometry heading increment to recover that frame.
            previous_heading = _wrap(self._heading - theta)
            travel_angle = previous_heading + alpha
            displacement = distance * np.array([np.cos(travel_angle), np.sin(travel_angle)])
            observed_previous_velocity = (
                displacement - POSITION_INPUT_GAIN * self._previous_command
            ) / POSITION_VELOCITY_GAIN
            observed_endpoint_velocity = (
                VELOCITY_DECAY * observed_previous_velocity + INPUT_GAIN * self._previous_command
            )
            self._velocity = (
                (1.0 - self.odometry_velocity_gain) * predicted_velocity
                + self.odometry_velocity_gain * observed_endpoint_velocity
            )
            # Odometry supplies an accurate relative displacement; blend it
            # with the model increment before the low-bandwidth GPS correction.
            predicted_increment = predicted_position - self._position
            self._position += (
                self.odometry_position_gain * displacement
                + (1.0 - self.odometry_position_gain) * predicted_increment
            )
        else:
            self._velocity = predicted_velocity
            self._position = predicted_position

        if gps_ok:
            self._position += self.gps_position_gain * (np.asarray(gps_position, dtype=float) - self._position)

        return StateEstimate(
            self._position.copy(), self._velocity.copy(), self._heading,
            odom_ok, gps_ok, compass_ok,
        )


class TruthStateInput:
    """Evaluation-only adapter; never use inside a competition controller."""

    @staticmethod
    def from_values(position: Sequence[float], velocity: Sequence[float], heading: float) -> StateEstimate:
        return StateEstimate(
            np.asarray(position, dtype=float).copy(),
            np.asarray(velocity, dtype=float).copy(),
            float(heading), False, False, False,
        )

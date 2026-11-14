"""World-frame velocity controllers based on the identified dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .dynamics import INPUT_GAIN, VELOCITY_DECAY


def limit_unit_circle(vector: Sequence[float]) -> tuple[np.ndarray, bool]:
    value = np.asarray(vector, dtype=float)
    if value.shape != (2,):
        raise ValueError("a two-dimensional vector is required")
    norm = float(np.linalg.norm(value))
    if norm > 1.0:
        return value / norm, True
    return value.copy(), False


def world_to_body(command_world: Sequence[float], heading: float) -> np.ndarray:
    x, y = np.asarray(command_world, dtype=float)
    cosine, sine = np.cos(heading), np.sin(heading)
    return np.array([cosine * x + sine * y, -sine * x + cosine * y])


def body_to_world(command_body: Sequence[float], heading: float) -> np.ndarray:
    forward, lateral = np.asarray(command_body, dtype=float)
    cosine, sine = np.cos(heading), np.sin(heading)
    return np.array([cosine * forward - sine * lateral, sine * forward + cosine * lateral])


@dataclass(frozen=True)
class VelocityControlOutput:
    command_world: np.ndarray
    raw_command_world: np.ndarray
    saturated: bool
    velocity_error: np.ndarray
    controller: str


class VelocityController(Protocol):
    name: str

    def reset(self) -> None: ...

    def compute(self, target_velocity: Sequence[float], measured_velocity: Sequence[float]) -> VelocityControlOutput: ...


class InverseDynamicsVelocityController:
    """One-step inverse model; no additional feedback term is stacked on it."""

    name = "inverse_dynamics"

    def __init__(self, velocity_decay: float = VELOCITY_DECAY, input_gain: float = INPUT_GAIN):
        self.velocity_decay = float(velocity_decay)
        self.input_gain = float(input_gain)
        if self.input_gain <= 0:
            raise ValueError("input_gain must be positive")

    def reset(self) -> None:
        return None

    def compute(self, target_velocity, measured_velocity) -> VelocityControlOutput:
        target = np.asarray(target_velocity, dtype=float)
        measured = np.asarray(measured_velocity, dtype=float)
        raw = (target - self.velocity_decay * measured) / self.input_gain
        command, saturated = limit_unit_circle(raw)
        return VelocityControlOutput(command, raw, saturated, target - measured, self.name)


class SteadyStateFeedbackVelocityController:
    """Steady-state feedforward plus proportional velocity feedback."""

    name = "steady_state_feedback"

    def __init__(
        self,
        *,
        closed_loop_pole: float = 0.6,
        velocity_decay: float = VELOCITY_DECAY,
        input_gain: float = INPUT_GAIN,
    ):
        if not -1.0 < closed_loop_pole < 1.0:
            raise ValueError("closed_loop_pole must be inside the unit circle")
        self.closed_loop_pole = float(closed_loop_pole)
        self.velocity_decay = float(velocity_decay)
        self.input_gain = float(input_gain)
        self.kp = (self.velocity_decay - self.closed_loop_pole) / self.input_gain
        self.steady_state_gain = (1.0 - self.velocity_decay) / self.input_gain

    def reset(self) -> None:
        return None

    def compute(self, target_velocity, measured_velocity) -> VelocityControlOutput:
        target = np.asarray(target_velocity, dtype=float)
        measured = np.asarray(measured_velocity, dtype=float)
        error = target - measured
        raw = self.steady_state_gain * target + self.kp * error
        command, saturated = limit_unit_circle(raw)
        return VelocityControlOutput(command, raw, saturated, error, self.name)

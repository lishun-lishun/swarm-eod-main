"""Discrete-model braking and low-speed precision point stopping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from .dynamics import predict_forward_braking_distance
from .velocity_controller import VelocityControlOutput, VelocityController, limit_unit_circle


class StopMode(str, Enum):
    CRUISE = "CRUISE"
    BRAKE = "BRAKE"
    APPROACH = "APPROACH"
    HOLD = "HOLD"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class PrecisionStopOutput:
    target_velocity: np.ndarray
    command_world: np.ndarray
    force_saturated: bool
    mode: StopMode
    predicted_stop_distance: float
    remaining_distance: float
    stable_frames: int


class PrecisionStopController:
    def __init__(
        self,
        velocity_controller: VelocityController,
        *,
        cruise_speed: float = 8.0,
        brake_margin: float = 0.35,
        approach_speed: float = 1.5,
        position_kp: float = 0.28,
        position_kd: float = 0.55,
        hold_position_tolerance: float = 0.30,
        hold_speed_tolerance: float = 0.08,
        stable_frames_required: int = 8,
    ):
        self.velocity_controller = velocity_controller
        self.cruise_speed = float(cruise_speed)
        self.brake_margin = float(brake_margin)
        self.approach_speed = float(approach_speed)
        self.position_kp = float(position_kp)
        self.position_kd = float(position_kd)
        self.hold_position_tolerance = float(hold_position_tolerance)
        self.hold_speed_tolerance = float(hold_speed_tolerance)
        self.stable_frames_required = int(stable_frames_required)
        self.reset()

    def reset(self) -> None:
        self.mode = StopMode.CRUISE
        self.stable_frames = 0
        self.velocity_controller.reset()

    def compute(self, position: Sequence[float], velocity: Sequence[float], target_position: Sequence[float]) -> PrecisionStopOutput:
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        target = np.asarray(target_position, dtype=float)
        error = target - position
        remaining = float(np.linalg.norm(error))
        direction = error / remaining if remaining > 1e-12 else np.zeros(2)
        closing_speed = max(0.0, float(velocity @ direction))
        prediction = predict_forward_braking_distance(closing_speed)

        if self.mode == StopMode.CRUISE and remaining <= prediction.distance + self.brake_margin:
            self.mode = StopMode.BRAKE

        if self.mode == StopMode.BRAKE:
            # High-speed stopping is an explicit active reverse command, not
            # zero thrust.  Preserve transverse damping by braking opposite the
            # full current velocity vector.
            speed = float(np.linalg.norm(velocity))
            command = -velocity / speed if speed > 1e-12 else np.zeros(2)
            command, saturated = limit_unit_circle(command)
            target_velocity = np.zeros(2)
            if closing_speed <= 0.75 or remaining <= 1.5:
                self.mode = StopMode.APPROACH
        else:
            if self.mode == StopMode.CRUISE:
                target_velocity = self.cruise_speed * direction
            else:
                target_velocity = self.position_kp * error - self.position_kd * velocity
                limit = 0.55 if self.mode in (StopMode.HOLD, StopMode.COMPLETE) else self.approach_speed
                norm = float(np.linalg.norm(target_velocity))
                if norm > limit:
                    target_velocity *= limit / norm
            control: VelocityControlOutput = self.velocity_controller.compute(target_velocity, velocity)
            command, saturated = control.command_world, control.saturated

        speed = float(np.linalg.norm(velocity))
        if self.mode in (StopMode.APPROACH, StopMode.HOLD, StopMode.COMPLETE):
            if remaining <= self.hold_position_tolerance and speed <= 0.16:
                self.mode = StopMode.HOLD
            elif self.mode in (StopMode.HOLD, StopMode.COMPLETE) and (
                remaining > 0.75 or speed > 0.45
            ):
                self.mode = StopMode.APPROACH

        if self.mode == StopMode.HOLD and remaining <= self.hold_position_tolerance and speed <= self.hold_speed_tolerance:
            self.stable_frames += 1
            if self.stable_frames >= self.stable_frames_required:
                self.mode = StopMode.COMPLETE
        else:
            self.stable_frames = 0

        return PrecisionStopOutput(
            target_velocity=np.asarray(target_velocity, dtype=float),
            command_world=np.asarray(command, dtype=float),
            force_saturated=bool(saturated),
            mode=self.mode,
            predicted_stop_distance=prediction.distance,
            remaining_distance=remaining,
            stable_frames=self.stable_frames,
        )

"""Predictive precision stop with separate motion and safety states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from .braking_safety import BoundaryConstraint, BrakingSafetyLayer
from .predictive_brake import BrakingMode
from .velocity_controller import VelocityController


class MotionMode(str, Enum):
    CRUISE = "CRUISE"
    APPROACH = "APPROACH"
    HOLD = "HOLD"
    COMPLETE = "COMPLETE"


StopModeV2 = MotionMode


@dataclass(frozen=True)
class PrecisionStopV2Output:
    target_velocity: np.ndarray
    nominal_thrust: np.ndarray
    safe_thrust: np.ndarray
    motion_mode: MotionMode
    safety_mode: BrakingMode
    braking: object
    stable_frames: int
    original_target: np.ndarray
    safe_target: np.ndarray
    target_inset: float

    @property
    def mode(self) -> MotionMode:
        return self.motion_mode


class PrecisionStopControllerV2:
    def __init__(
        self,
        velocity_controller: VelocityController,
        *,
        cruise_speed: float = 10.0,
        approach_speed: float = 1.2,
        position_kp: float = 0.28,
        position_kd: float = 0.55,
        hold_position_tolerance: float = 1.0,
        hold_speed_tolerance: float = 0.15,
        stable_frames_required: int = 8,
        safety_margin: float = 0.0,
        velocity_confidence: float = 0.0,
        position_confidence: float = 0.0,
        target_reserve: float = 2.0,
    ):
        self.velocity_controller = velocity_controller
        self.safety = BrakingSafetyLayer()
        self.cruise_speed = float(cruise_speed)
        self.approach_speed = float(approach_speed)
        self.position_kp = float(position_kp)
        self.position_kd = float(position_kd)
        self.hold_position_tolerance = float(hold_position_tolerance)
        self.hold_speed_tolerance = float(hold_speed_tolerance)
        self.stable_frames_required = int(stable_frames_required)
        self.safety_margin = float(safety_margin)
        self.velocity_confidence = float(velocity_confidence)
        self.position_confidence = float(position_confidence)
        self.target_reserve = float(target_reserve)
        self.reset()

    def reset(self) -> None:
        self.motion_mode = MotionMode.CRUISE
        self.safety_mode = BrakingMode.NORMAL
        self.stable_frames = 0
        self._complete_latched = False
        self._original_target: Optional[np.ndarray] = None
        self._safe_target: Optional[np.ndarray] = None
        self._target_inset = 0.0
        self._boundary_signature = None
        self.velocity_controller.reset()
        self.safety.reset()

    def start_new_target(
        self,
        target_position: Sequence[float],
        boundary: BoundaryConstraint,
        *,
        position_uncertainty: float = 0.0,
        velocity_uncertainty: float = 0.0,
    ) -> np.ndarray:
        """Latch a feasible target once; never jitter it with frame noise."""
        self.motion_mode = MotionMode.CRUISE
        self.safety_mode = BrakingMode.NORMAL
        self.stable_frames = 0
        self._complete_latched = False
        self.velocity_controller.reset()
        self.safety.reset()
        target = np.asarray(target_position, dtype=float).copy()
        epsilon_p = self.position_confidence * max(0.0, float(position_uncertainty))
        epsilon_v = self.velocity_confidence * max(0.0, float(velocity_uncertainty))
        target_distance = (
            epsilon_p
            + self.safety.predictive.stopping_model.stopping_distance(epsilon_v)
            + self.safety_margin
            + self.target_reserve
        )
        excess = max(0.0, float(boundary.normal @ target - boundary.offset + target_distance))
        self._original_target = target
        self._safe_target = target - excess * boundary.normal
        self._target_inset = excess
        self._boundary_signature = (tuple(boundary.normal), float(boundary.offset))
        return self._safe_target.copy()

    def _target_changed(self, target: np.ndarray, boundary: BoundaryConstraint) -> bool:
        signature = (tuple(boundary.normal), float(boundary.offset))
        return (
            self._original_target is None
            or not np.allclose(target, self._original_target, atol=1e-12, rtol=0.0)
            or signature != self._boundary_signature
        )

    def compute(
        self,
        *,
        position: Sequence[float],
        velocity: Sequence[float],
        target_position: Sequence[float],
        boundary: BoundaryConstraint,
        position_uncertainty: float = 0.0,
        velocity_uncertainty: float = 0.0,
    ) -> PrecisionStopV2Output:
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        original_target = np.asarray(target_position, dtype=float)
        if self._target_changed(original_target, boundary):
            self.start_new_target(
                original_target, boundary,
                position_uncertainty=position_uncertainty,
                velocity_uncertainty=velocity_uncertainty,
            )
        safe_target = self._safe_target.copy()
        error = safe_target - position
        distance = float(np.linalg.norm(error))
        speed = float(np.linalg.norm(velocity))
        direction = error / distance if distance > 1e-12 else np.zeros(2)

        if self._complete_latched:
            target_velocity = np.zeros(2)
        elif self.motion_mode in (MotionMode.APPROACH, MotionMode.HOLD):
            target_velocity = self.position_kp * error - self.position_kd * velocity
            limit = 0.5 if self.motion_mode == MotionMode.HOLD else self.approach_speed
            norm = float(np.linalg.norm(target_velocity))
            if norm > limit:
                target_velocity *= limit / norm
        else:
            target_velocity = self.cruise_speed * direction

        nominal = self.velocity_controller.compute(target_velocity, velocity).command_world
        safe = self.safety.compute(
            position=position, velocity=velocity, nominal_thrust=nominal,
            boundary=boundary, safety_margin=self.safety_margin,
            velocity_uncertainty=velocity_uncertainty,
            position_uncertainty=position_uncertainty,
            velocity_confidence=self.velocity_confidence,
            position_confidence=self.position_confidence,
        )
        self.safety_mode = safe.scalar.mode

        if not self._complete_latched:
            if distance <= self.hold_position_tolerance and speed <= 0.25:
                self.motion_mode = MotionMode.HOLD
            elif self.motion_mode == MotionMode.HOLD and (distance > 1.5 or speed > 0.45):
                self.motion_mode = MotionMode.APPROACH
            elif distance <= 4.0 and speed <= 1.5:
                self.motion_mode = MotionMode.APPROACH
            elif self.motion_mode not in (MotionMode.APPROACH, MotionMode.HOLD):
                self.motion_mode = MotionMode.CRUISE

            completion_condition = (
                distance <= self.hold_position_tolerance
                and speed <= self.hold_speed_tolerance
                and safe.scalar.safety_residual <= 1e-9
                and safe.scalar.feasible
            )
            if completion_condition:
                self.stable_frames += 1
                if self.stable_frames >= self.stable_frames_required:
                    self._complete_latched = True
                    self.motion_mode = MotionMode.COMPLETE
            else:
                self.stable_frames = 0
        else:
            self.motion_mode = MotionMode.COMPLETE

        return PrecisionStopV2Output(
            target_velocity, nominal, safe.command_world, self.motion_mode,
            self.safety_mode, safe, self.stable_frames,
            self._original_target.copy(), safe_target, self._target_inset,
        )

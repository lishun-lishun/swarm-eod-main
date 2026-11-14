"""One-step predictive stopping constraint and bounded scalar solver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time

import numpy as np

from .dynamics import INPUT_GAIN, POSITION_INPUT_GAIN, POSITION_VELOCITY_GAIN, VELOCITY_DECAY
from .nonlinear_braking import NonlinearStoppingModel


class BrakingMode(str, Enum):
    NORMAL = "NORMAL"
    PREDICTIVE_BRAKE = "PREDICTIVE_BRAKE"
    INFEASIBLE_BRAKING = "INFEASIBLE_BRAKING"


@dataclass(frozen=True)
class PredictiveBrakeOutput:
    safe_thrust: float
    nominal_thrust: float
    mode: BrakingMode
    predicted_next_velocity: float
    predicted_next_displacement: float
    current_frame_max_displacement: float
    nonlinear_stopping_distance: float
    total_forward_excursion: float
    required_stopping_distance: float
    safety_margin: float
    safety_residual: float
    feasible: bool
    bisection_iterations: int
    compute_time_ns: int


class PredictiveBrakingController:
    def __init__(
        self, stopping_model: NonlinearStoppingModel | None = None, *,
        bisection_iterations: int = 20, recovery_rate: float = 0.2,
    ):
        self.stopping_model = stopping_model or NonlinearStoppingModel()
        self.bisection_iterations = int(bisection_iterations)
        self.recovery_rate = float(recovery_rate)
        self._last_safe_thrust = None

    def reset(self) -> None:
        self._last_safe_thrust = None

    def candidate_prediction(self, speed: float, thrust: float) -> tuple[float, float, float, float, float]:
        thrust = float(np.clip(thrust, -1.0, 1.0))
        next_velocity = VELOCITY_DECAY * float(speed) + INPUT_GAIN * thrust
        frame_end = POSITION_VELOCITY_GAIN * float(speed) + POSITION_INPUT_GAIN * thrust
        frame_peak = self.stopping_model.frame_max_displacement(speed, thrust)
        future = self.stopping_model.stopping_distance(max(0.0, next_velocity))
        total_peak = max(frame_peak, frame_end + future)
        return frame_end, next_velocity, frame_peak, future, total_peak

    def safety_residual(self, speed: float, thrust: float, remaining_distance: float, safety_margin: float) -> float:
        _, _, _, _, total_peak = self.candidate_prediction(speed, thrust)
        return total_peak + float(safety_margin) - float(remaining_distance)

    def compute(
        self,
        *,
        speed: float,
        nominal_thrust: float,
        remaining_distance: float,
        safety_margin: float = 0.0,
        velocity_uncertainty: float = 0.0,
        position_uncertainty: float = 0.0,
        velocity_confidence: float = 0.0,
        position_confidence: float = 0.0,
    ) -> PredictiveBrakeOutput:
        started = time.perf_counter_ns()
        conservative_speed = max(0.0, float(speed) + velocity_confidence * max(0.0, velocity_uncertainty))
        conservative_remaining = (
            float(remaining_distance)
            - position_confidence * max(0.0, position_uncertainty)
        )
        nominal = float(np.clip(nominal_thrust, -1.0, 1.0))
        residual_min = self.safety_residual(conservative_speed, -1.0, conservative_remaining, safety_margin)
        iterations = 0
        if residual_min > 1e-10:
            safe = -1.0
            mode = BrakingMode.INFEASIBLE_BRAKING
            feasible = False
        else:
            residual_nominal = self.safety_residual(
                conservative_speed, nominal, conservative_remaining, safety_margin
            )
            if residual_nominal <= 1e-10:
                safe = nominal
                mode = BrakingMode.NORMAL
            else:
                lo, hi = -1.0, nominal
                for _ in range(self.bisection_iterations):
                    iterations += 1
                    mid = 0.5 * (lo + hi)
                    if self.safety_residual(
                        conservative_speed, mid, conservative_remaining, safety_margin
                    ) <= 0.0:
                        lo = mid
                    else:
                        hi = mid
                safe = lo
                mode = BrakingMode.PREDICTIVE_BRAKE
            feasible = True

        # Restriction is immediate, while release is rate-limited.  This
        # one-sided hysteresis cannot make a safe command less safe and avoids
        # alternating brake/accelerate commands on noisy boundary estimates.
        if (
            feasible and self._last_safe_thrust is not None
            and safe > self._last_safe_thrust + self.recovery_rate
        ):
            safe = self._last_safe_thrust + self.recovery_rate
            mode = BrakingMode.PREDICTIVE_BRAKE
        self._last_safe_thrust = safe
        frame_end, next_velocity, frame_peak, future, total_peak = self.candidate_prediction(
            conservative_speed, safe
        )
        required = total_peak + safety_margin
        residual = required - conservative_remaining
        return PredictiveBrakeOutput(
            safe, nominal, mode, next_velocity, frame_end, frame_peak, future,
            total_peak, required,
            float(safety_margin), residual, feasible, iterations,
            time.perf_counter_ns() - started,
        )

"""Single-boundary two-dimensional thrust allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .predictive_brake import PredictiveBrakeOutput, PredictiveBrakingController


@dataclass(frozen=True)
class BoundaryConstraint:
    normal: np.ndarray
    offset: float

    def __post_init__(self):
        normal = np.asarray(self.normal, dtype=float)
        norm = float(np.linalg.norm(normal))
        if normal.shape != (2,) or norm <= 0:
            raise ValueError("boundary normal must be a nonzero 2-vector")
        object.__setattr__(self, "normal", normal / norm)


@dataclass(frozen=True)
class BrakingSafetyOutput:
    command_world: np.ndarray
    normal_thrust: float
    tangential_thrust: float
    remaining_distance: float
    scalar: PredictiveBrakeOutput


class BrakingSafetyLayer:
    """Prioritise normal braking, then use the remaining unit-circle budget."""

    def __init__(self, predictive: PredictiveBrakingController | None = None):
        self.predictive = predictive or PredictiveBrakingController()

    def reset(self) -> None:
        self.predictive.reset()

    def compute(
        self,
        *,
        position: Sequence[float],
        velocity: Sequence[float],
        nominal_thrust: Sequence[float],
        boundary: BoundaryConstraint,
        safety_margin: float = 0.0,
        velocity_uncertainty: float = 0.0,
        position_uncertainty: float = 0.0,
        velocity_confidence: float = 0.0,
        position_confidence: float = 0.0,
    ) -> BrakingSafetyOutput:
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        nominal = np.asarray(nominal_thrust, dtype=float)
        normal = boundary.normal
        tangent = np.array([-normal[1], normal[0]])
        remaining = float(boundary.offset - position @ normal)
        nominal_normal = float(nominal @ normal)
        nominal_tangent = float(nominal @ tangent)
        scalar = self.predictive.compute(
            speed=float(velocity @ normal), nominal_thrust=nominal_normal,
            remaining_distance=remaining, safety_margin=safety_margin,
            velocity_uncertainty=velocity_uncertainty,
            position_uncertainty=position_uncertainty,
            velocity_confidence=velocity_confidence,
            position_confidence=position_confidence,
        )
        normal_thrust = scalar.safe_thrust
        tangent_limit = float(np.sqrt(max(0.0, 1.0 - normal_thrust**2)))
        tangent_thrust = float(np.clip(nominal_tangent, -tangent_limit, tangent_limit))
        command = normal_thrust * normal + tangent_thrust * tangent
        return BrakingSafetyOutput(command, normal_thrust, tangent_thrust, remaining, scalar)

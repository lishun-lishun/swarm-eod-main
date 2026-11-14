"""Stage-2 composition layer: estimator -> target -> one velocity controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .precision_stop import PrecisionStopController, PrecisionStopOutput
from .velocity_controller import VelocityController, world_to_body
from .velocity_estimator import LegalVelocityEstimator, StateEstimate


@dataclass(frozen=True)
class FlightCommand:
    forward: float
    lateral: float
    rotation: float
    world_command: np.ndarray
    estimate: StateEstimate
    stop: PrecisionStopOutput


class Stage2FlightController:
    """Legal-sensor precision-stop controller suitable for later composition."""

    def __init__(self, velocity_controller: VelocityController, **stop_kwargs):
        self.estimator = LegalVelocityEstimator()
        self.stop_controller = PrecisionStopController(velocity_controller, **stop_kwargs)

    def reset(self) -> None:
        self.estimator.reset()
        self.stop_controller.reset()

    def compute_stop_command(
        self,
        *,
        target_position: Sequence[float],
        odometry: Optional[Sequence[float]],
        compass: Optional[float],
        gps_position: Optional[Sequence[float]],
    ) -> FlightCommand:
        estimate = self.estimator.update(odometry=odometry, compass=compass, gps_position=gps_position)
        stop = self.stop_controller.compute(estimate.position, estimate.velocity, target_position)
        body = world_to_body(stop.command_world, estimate.heading)
        self.estimator.set_previous_command(stop.command_world)
        return FlightCommand(float(body[0]), float(body[1]), 0.0, stop.command_world.copy(), estimate, stop)

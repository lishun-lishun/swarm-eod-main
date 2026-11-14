"""Legal localization wrapper with explicit degraded-mode uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .velocity_estimator import LegalVelocityEstimator, StateEstimate


@dataclass(frozen=True)
class LocalizationOutput:
    estimate: StateEstimate
    position_sigma: float
    velocity_sigma: float
    gps_outage_frames: int
    degraded: bool


class RobustLocalization:
    def __init__(self):
        self.estimator = LegalVelocityEstimator(odometry_position_gain=0.2)
        self.gps_outage = 0; self.position_sigma = 5.0; self.velocity_sigma = 0.45

    def reset(self) -> None:
        self.estimator.reset(); self.gps_outage = 0
        self.position_sigma = 5.0; self.velocity_sigma = 0.45

    def set_previous_command(self, command: Sequence[float], rotation: float = 0.0) -> None:
        self.estimator.set_previous_command(command, rotation)

    def update(self, *, odometry, compass, gps_position) -> LocalizationOutput:
        estimate = self.estimator.update(odometry=odometry, compass=compass, gps_position=gps_position)
        if estimate.used_gps:
            self.gps_outage = 0
            self.position_sigma = max(3.5, 0.82 * self.position_sigma)
        else:
            self.gps_outage += 1
            self.position_sigma = min(35.0, self.position_sigma + 0.08)
        self.velocity_sigma = min(1.2, 0.45 + 0.003 * self.gps_outage)
        return LocalizationOutput(estimate, self.position_sigma, self.velocity_sigma,
                                  self.gps_outage, self.gps_outage > 8)



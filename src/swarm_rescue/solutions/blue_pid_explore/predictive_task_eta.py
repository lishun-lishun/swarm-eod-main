"""Local completion-time estimates for legally discovered bomb tasks."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TripEstimate:
    approach_frames: float
    grasp_frames: float
    return_frames: float
    queue_frames: float
    risk_frames: float
    uncertain_home: bool

    @property
    def total_frames(self) -> float:
        return (self.approach_frames + self.grasp_frames + self.return_frames
                + self.queue_frames + self.risk_frames)


@dataclass(frozen=True)
class ETASettings:
    cruise_empty_px_per_frame: float = 7.0
    cruise_loaded_px_per_frame: float = 2.6
    approach_multiplier: float = 1.20
    loaded_multiplier: float = 1.25
    nominal_grasp_frames: float = 35.0
    nominal_queue_frames: float = 20.0
    unknown_center_return_px: float = 650.0
    unknown_center_risk_frames: float = 90.0
    failed_attempt_risk_frames: float = 45.0


class CompletionTimeEstimator:
    def __init__(self, settings: ETASettings | None = None):
        self.settings = settings or ETASettings()

    def estimate(
        self,
        approach_path_px: float,
        loaded_home_path_px: float | None,
        *,
        failure_count: int = 0,
        queue_frames: float | None = None,
        approach_risk_frames: float = 0.0,
        center_observation_distance_px: float | None = None,
        center_field_is_terminal: bool = False,
    ) -> TripEstimate:
        s = self.settings
        if not math.isfinite(approach_path_px) or approach_path_px < 0:
            return TripEstimate(math.inf, 0, 0, 0, 0, False)
        uncertain = (loaded_home_path_px is None
                     or not math.isfinite(loaded_home_path_px))
        home = (s.unknown_center_return_px if uncertain
                else max(0.0, float(loaded_home_path_px)))
        if (center_field_is_terminal
                and center_observation_distance_px is not None
                and math.isfinite(center_observation_distance_px)):
            home = max(home, max(0.0, center_observation_distance_px - 25.0))
        return TripEstimate(
            approach_frames=s.approach_multiplier * approach_path_px
            / s.cruise_empty_px_per_frame,
            grasp_frames=s.nominal_grasp_frames,
            return_frames=s.loaded_multiplier * home
            / s.cruise_loaded_px_per_frame,
            queue_frames=(s.nominal_queue_frames if queue_frames is None
                          else max(0.0, queue_frames)),
            risk_frames=(max(0, failure_count) * s.failed_attempt_risk_frames
                         + max(0.0, approach_risk_frames)
                         + (s.unknown_center_risk_frames if uncertain else 0.0)),
            uncertain_home=uncertain,
        )


class AssignmentHysteresis:
    def __init__(self, *, min_gain_frames: float = 35.0,
                 fractional_gain: float = 0.15):
        self.min_gain_frames = float(min_gain_frames)
        self.fractional_gain = float(fractional_gain)

    def should_switch(self, incumbent_eta: float | None,
                      challenger_eta: float, *,
                      incumbent_still_valid: bool = True) -> bool:
        if not math.isfinite(challenger_eta):
            return False
        if (not incumbent_still_valid or incumbent_eta is None
                or not math.isfinite(incumbent_eta)):
            return True
        required = max(self.min_gain_frames,
                       self.fractional_gain * incumbent_eta)
        return incumbent_eta - challenger_eta >= required

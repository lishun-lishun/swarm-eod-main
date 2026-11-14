"""V6 discrete stopping envelope for verified EMPTY-body dynamics only.

Pure calibration/reference code. It is NOT a loaded-body model, does NOT
replace the existing Pymunk-validated predictive safety filter, and provides
only conservative time-step upper bounds for head-on braking.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ScalarDynamics:
    decay: float = 0.95
    input_gain: float = 0.5729312289729012

    def __post_init__(self):
        if not 0.0 < self.decay < 1.0 or self.input_gain <= 0:
            raise ValueError('invalid dynamics')


def stopping_distance_upper(speed_px_per_frame: float,
                            *, model=ScalarDynamics(), max_steps=300) -> float:
    """Upper bound by summing pre-brake velocities of control intervals.

    Assumes max opposing unit-disk thrust fully available in this axis.
    This bound becomes invalid if longitudinal thrust is simultaneously
    reserved for turning or lateral motion: external safety layer must account
    for the actual allowed normal-axis deceleration.
    """
    v=max(0.0,float(speed_px_per_frame))
    if not math.isfinite(v):
        return math.inf
    d=0.0
    for _ in range(max_steps):
        if v<=0:
            break
        d+=v
        v=max(0.0,model.decay*v-model.input_gain)
    return d


def max_certified_speed(gap_px:float, *, margin_px:float,
                        vmax_px_per_frame=11.46, model=ScalarDynamics()):
    """Invert the *empty* discrete stop bound; never apply to cargo."""
    available=float(gap_px)-float(margin_px)
    if not math.isfinite(available) or available<=0:
        return 0.0
    low,high=0.0,float(vmax_px_per_frame)
    for _ in range(45):
        mid=(low+high)/2
        if stopping_distance_upper(mid,model=model)<=available:
            low=mid
        else:
            high=mid
    return low

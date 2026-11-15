"""Discrete stopping-distance shadow audit (not loaded motor control).

v[k+1] = a*v[k] + b*u[k]. For an emergency opposite thrust u=-1,
iterate the one-dimensional speed until nonpositive. Includes reaction steps.
Cargo use requires an independently validated conservative lower-bound b.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BrakeModel:
    decay:float
    opposite_thrust_gain:float
    def __post_init__(self):
        if not (0<self.decay<1 and self.opposite_thrust_gain>0):
            raise ValueError('need stable strictly positive braking model')


EMPTY=BrakeModel(0.95,0.5729312289729012)


def discrete_stopping_distance(speed:float, model:BrakeModel=EMPTY,
                              *, reaction_frames:int=2, max_frames:int=120)->float:
    if not math.isfinite(speed) or speed<0 or reaction_frames<0 or max_frames<1:
        raise ValueError('bad stopping query')
    v=float(speed); traveled=0.0
    for _ in range(int(reaction_frames)):
        v*=model.decay
        traveled+=v
    for _ in range(int(max_frames)):
        v=max(0.,model.decay*v-model.opposite_thrust_gain)
        traveled+=v
        if v<=0:
            return traveled
    return float('inf')


def cap_speed_for_clearance(clearance_px:float, model:BrakeModel=EMPTY,
                            *, margin_px:float=8., reaction_frames:int=2,
                            physical_speed_ceiling:float=11.4)->float:
    if not math.isfinite(clearance_px) or clearance_px<0:
        raise ValueError('bad clearance')
    gap=max(0.0,clearance_px-margin_px)
    if gap<=0:
        return 0.0
    lo,hi=0.0,float(physical_speed_ceiling)
    for _ in range(35):
        m=.5*(lo+hi)
        if discrete_stopping_distance(m,model,reaction_frames=reaction_frames)<=gap:
            lo=m
        else:
            hi=m
    return lo

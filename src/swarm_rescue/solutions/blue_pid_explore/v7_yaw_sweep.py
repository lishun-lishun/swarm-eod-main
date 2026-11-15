"""Optional small yaw phase sweep AFTER reaching a physically valid viewpoint.

35 official rays span 360 degrees with endpoint duplication; ray spacing
is 2*pi/(35-1). Three relative yaw phases interleave their directions.
This does not see through occluders and never asserts 'no bomb'.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


def wrap(a:float)->float:
    return (float(a)+math.pi)%(2*math.pi)-math.pi


@dataclass(frozen=True)
class SweepDecision:
    rotation:float
    done:bool
    phase:int
    target_heading:float


class YawPhaseSweep:
    def __init__(self, *, ray_count:int=35, phases:int=3,
                 tolerance_rad:float=0.023, gain:float=2.5, max_rotation:float=0.35):
        if ray_count<3 or phases<2 or tolerance_rad<=0 or gain<=0 or max_rotation<=0:
            raise ValueError('invalid sweep settings')
        self.ray_spacing=2*math.pi/(ray_count-1)
        self.phases=int(phases)
        self.tol=float(tolerance_rad)
        self.gain=float(gain)
        self.max_rotation=float(max_rotation)
        self.reset()

    def reset(self):
        self.base=None
        self.phase=0

    def start(self, current_heading:float):
        self.base=wrap(current_heading)
        self.phase=0

    def update(self, *, current_heading:float, carrying:bool,
               visible_target:bool, stable_standpoint:bool)->SweepDecision:
        if carrying or visible_target or not stable_standpoint:
            self.reset()
            return SweepDecision(0.,True,0,wrap(current_heading))
        if self.base is None:
            self.start(current_heading)
        if self.phase>=self.phases:
            return SweepDecision(0.,True,self.phase,wrap(current_heading))
        target=wrap(self.base+self.ray_spacing*self.phase/self.phases)
        error=wrap(target-current_heading)
        if abs(error)<=self.tol:
            self.phase+=1
            if self.phase>=self.phases:
                return SweepDecision(0.,True,self.phase,target)
            target=wrap(self.base+self.ray_spacing*self.phase/self.phases)
            error=wrap(target-current_heading)
        command=max(-self.max_rotation,min(self.max_rotation,self.gain*error))
        return SweepDecision(command,False,self.phase,target)

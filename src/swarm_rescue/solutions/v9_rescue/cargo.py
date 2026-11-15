"""Direction-dependent BOUNDING proxy for a drone + PivotJoint bomb.

This is not a rigid-body dynamics model. The bomb's pose after pickup is not
available through the legal sensor API. Bounds are heuristic and must be
validated against offline true Pymunk cargo trajectories before deployment.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .geometry import wrap

@dataclass(frozen=True)
class CargoExtent:
    along_px: float
    lateral_px: float
    yaw_clearance_required_px: float
    uncertainty_rad: float


def direction_dependent_extent(*,movement_relative_yaw:float,
                               cargo_bearing_body:float|None,
                               uncertainty_rad:float=0.8,
                               drone_radius_px:float=18.0,
                               grasp_center_px:float=27.0,
                               bomb_radius_px:float=12.0)->CargoExtent:
    """Approximate swept projected footprint for a near-constant heading.

    When bearing is unknown, assume 360-degree reach and do not claim a narrow
    passage is safe. Pass this only to preflight velocity selection; live LiDAR
    and official simulator collisions remain authoritative.
    """
    if cargo_bearing_body is None or not math.isfinite(cargo_bearing_body):
        r=grasp_center_px+bomb_radius_px
        return CargoExtent(max(drone_radius_px,r),max(drone_radius_px,r),r,math.pi)
    spread=min(math.pi,max(0.,float(uncertainty_rad)))
    angles=np.linspace(-spread,spread,11)
    relative=np.array([wrap(cargo_bearing_body+d-movement_relative_yaw) for d in angles])
    ahead=max(0.,float(np.max(grasp_center_px*np.cos(relative)+bomb_radius_px)))
    width=float(np.max(np.abs(grasp_center_px*np.sin(relative))))+bomb_radius_px
    along=max(drone_radius_px,ahead)
    lateral=max(drone_radius_px,width)
    return CargoExtent(along,lateral,grasp_center_px+bomb_radius_px,spread)

class CargoSwingTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.bearing=None
        self.uncertainty=math.pi
        self.last_yaw=None
        self.last_carry=False
    def observe(self,*,carrying:bool,body_yaw:float,pickup_bearing:float|None=None):
        if carrying and not self.last_carry:
            self.bearing=pickup_bearing
            self.uncertainty=.55 if pickup_bearing is not None else math.pi
            self.last_yaw=float(body_yaw)
        elif carrying and self.last_yaw is not None:
            delta=abs(wrap(body_yaw-self.last_yaw))
            # Without observing bomb angle, uncertainty only grows.
            self.uncertainty=min(math.pi,self.uncertainty+.003+1.4*delta)
            self.last_yaw=float(body_yaw)
        if not carrying and self.last_carry:self.reset()
        self.last_carry=bool(carrying)
        return self.bearing,self.uncertainty

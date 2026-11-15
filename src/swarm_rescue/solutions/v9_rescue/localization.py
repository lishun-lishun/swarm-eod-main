"""Persistent ego-local coordinate frame; optional GPS anchoring and scan matching.

The grid and breadcrumbs always live in the same LOCAL frame. GPS merely
establishes a transform to/from world messages; an outage does not invalidate
local routes. Scan matching uses a PREVIOUSLY built static occupancy map.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from .geometry import Pose, rotate, wrap, valid_odom

@dataclass(frozen=True)
class LocalizationStatus:
    pose: Pose
    velocity: tuple[float, float]  # local px/control-frame
    sigma_xy: float
    sigma_yaw: float
    world_anchored: bool
    gps_active: bool
    source: str

class EgoLocalizer:
    def __init__(self):
        self.pose = Pose(0.0, 0.0, 0.0)
        self.velocity = np.zeros(2)
        self.world_origin = None
        self.world_yaw = None
        self.step = -1
        self.sigma_xy = 3.0
        self.sigma_yaw = math.radians(4.0)
        self.gps_active = False
        self.source = 'BOOT'
        self._last_odom = None
        self._last_gps_step = -1

    @property
    def anchored(self):
        return self.world_origin is not None and self.world_yaw is not None

    def world(self, xy=None):
        if not self.anchored:
            return None
        p = self.pose.xy() if xy is None else np.asarray(xy, dtype=float)
        return rotate(p, self.world_yaw) + self.world_origin

    def local_from_world(self, xy):
        if not self.anchored:
            return None
        return rotate(np.asarray(xy, dtype=float) - self.world_origin, -self.world_yaw)

    def predict(self, *, step, odom=None, gps=None, compass=None) -> LocalizationStatus:
        step = int(step)
        if step <= self.step:
            return self.status()
        previous = self.pose
        if valid_odom(odom):
            d, alpha, dtheta = np.asarray(odom, dtype=float)
            moved = rotate((d*math.cos(alpha), d*math.sin(alpha)), previous.yaw)
            self.pose = previous.plus(moved[0], moved[1], dtheta)
            self.velocity = moved.copy()
            self.sigma_xy = min(300.0, math.hypot(self.sigma_xy, 0.35+abs(d)*0.035))
            self.sigma_yaw = min(math.pi, math.hypot(self.sigma_yaw, math.radians(0.55)))
            self.source = 'ODOM'
        else:
            self.velocity[:] = 0.0
            self.sigma_xy = min(300., self.sigma_xy + 1.5)
            self.source = 'NO_ODOM'
        self.step = step
        self._last_odom = odom
        compass_ok = compass is not None and math.isfinite(float(compass))
        gps_ok = gps is not None and np.asarray(gps).shape == (2,) and np.isfinite(gps).all()
        self.gps_active = bool(gps_ok)
        if self.world_yaw is None and compass_ok:
            self.world_yaw = wrap(float(compass)-self.pose.yaw)
        if self.world_origin is None and gps_ok and self.world_yaw is not None:
            self.world_origin = np.asarray(gps, dtype=float) - rotate(self.pose.xy(), self.world_yaw)
            self.sigma_xy = min(self.sigma_xy, 5.0)
        if self.anchored and compass_ok:
            local_compass = wrap(float(compass) - self.world_yaw)
            error = wrap(local_compass-self.pose.yaw)
            if abs(error) < math.radians(50):
                self.pose = Pose(self.pose.x, self.pose.y, wrap(self.pose.yaw+0.22*error))
                self.sigma_yaw = max(math.radians(0.8), self.sigma_yaw*.82)
        if self.anchored and gps_ok:
            local_gps = self.local_from_world(gps)
            residual = local_gps-self.pose.xy()
            # Reject gross innovation caused by delayed/outlier GPS.
            if float(np.linalg.norm(residual)) < max(35.0, 5.0*self.sigma_xy):
                limited = np.clip(residual, -7.0, 7.0)
                self.pose = self.pose.plus(0.18*limited[0], 0.18*limited[1])
                self.sigma_xy = max(1.5, self.sigma_xy*0.82)
                self._last_gps_step = step
                self.source = 'ODOM_GPS'
        return self.status()

    def accept_scan_match(self, *, dx, dy, dyaw, improvement,
                          static_correspondences, max_translation=7.0):
        """Apply a bounded correction to the local frame. Never rewrites map."""
        if static_correspondences < 15 or improvement < 0.12:
            return False
        if (math.hypot(dx,dy)>max_translation or abs(dyaw)>math.radians(3.5)):
            return False
        self.pose = self.pose.plus(dx,dy,dyaw)
        self.sigma_xy = max(1.5, self.sigma_xy*.85)
        self.sigma_yaw = max(math.radians(.8), self.sigma_yaw*.92)
        self.source = 'ODOM_SCAN'
        return True

    def status(self):
        return LocalizationStatus(
            self.pose, tuple(float(v) for v in self.velocity),
            float(self.sigma_xy),float(self.sigma_yaw),
            self.anchored, bool(self.gps_active),self.source)


def correlative_scan_match(belief, predicted_pose, ranges, angles,
                           *, max_range=230.0, min_hits=15):
    """Conservative local scan-to-static-map match, using old occupied cells.

    Uses a tiny 5 x 5 x 3 pose search on a downsample of LiDAR endpoints.
    Returns (dx,dy,dyaw,normalized_improvement,correspondences) or None.
    It is deliberately NOT a full SLAM replacement or loop-closure algorithm.
    """
    rr=np.asarray(ranges,dtype=float)
    aa=np.asarray(angles,dtype=float)
    if rr.ndim!=1 or rr.shape!=aa.shape or rr.size<30:
        return None
    ids=np.arange(0,len(rr),max(1,len(rr)//45))
    sel=np.isfinite(rr[ids]) & (rr[ids]>5.0) & (rr[ids]<max_range-6.0)
    dist=rr[ids][sel]; ang=aa[ids][sel]
    if dist.size < min_hits:
        return None
    xy=predicted_pose.xy()
    occup=belief.log_odds > belief.occ_threshold
    if int(np.count_nonzero(occup)) < 45:
        return None
    def score(offx,offy,dyaw):
        theta=predicted_pose.yaw+dyaw+ang
        ex=xy[0]+offx+dist*np.cos(theta)
        ey=xy[1]+offy+dist*np.sin(theta)
        ix=np.floor((ex-belief.origin[0])/belief.cell_px).astype(int)
        iy=np.floor((ey-belief.origin[1])/belief.cell_px).astype(int)
        good=(ix>=0)&(iy>=0)&(ix<belief.w)&(iy<belief.h)
        if np.count_nonzero(good)<min_hits:
            return -math.inf,0
        hits=int(np.count_nonzero(occup[iy[good],ix[good]]))
        return hits/max(1,int(np.count_nonzero(good))),hits
    baseline,_=score(0.,0.,0.)
    if not math.isfinite(baseline):
        return None
    best=(-math.inf,0.,0.,0.,0)
    for x in (-6.,-3.,0.,3.,6.):
        for y in (-6.,-3.,0.,3.,6.):
            for th in (math.radians(-2.),0.,math.radians(2.)):
                val,hits=score(x,y,th)
                # Prior regularization prevents 6px repeated systematic drift.
                objective=val-.006*(abs(x)+abs(y))-.003*abs(math.degrees(th))
                if objective>best[0]:
                    best=(objective,x,y,th,hits)
    if best[-1]<min_hits:
        return None
    improvement=(best[0]-baseline)
    if improvement<0.12:
        return None
    return best[1],best[2],best[3],improvement,best[4]

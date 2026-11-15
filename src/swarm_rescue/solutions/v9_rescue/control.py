"""Single nominal motion generator + one local safety stage + action transform.

The loaded model is an empirical bounded controller, NOT an inverse of the
PivotJoint-coupled physical system; tune by official loaded Pymunk experiments.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from .geometry import Pose, RelativeDetection, wrap, rotate, to_body, clip_disk, norm, normalize
from .config import V9Config
from .cargo import direction_dependent_extent

@dataclass(frozen=True)
class MotionResult:
    forward: float
    lateral: float
    rotation: float
    grasper: int
    reason: str
    nominal_speed: float
    safe_speed: float

class ModelAwareMotion:
    def __init__(self,config:V9Config|None=None):
        self.cfg=config or V9Config()
        self.last_control=np.zeros(2)
        self.last_obstacle_reason='INIT'

    def braking_speed(self,gap_px,carrying):
        a=(self.cfg.loaded_decel_px_frame2 if carrying else self.cfg.empty_decel_px_frame2)
        return math.sqrt(max(0.,2.*a*max(0.,gap_px)))

    def _safe_direction(self,direction_local,desired_speed,lidar_ranges,lidar_angles,
                        *,carrying,semantic_target=None,near_target=False,
                        cargo_bearing=None,cargo_uncertainty=math.pi):
        rr=np.asarray(lidar_ranges,dtype=float)
        aa=np.asarray(lidar_angles,dtype=float)
        if rr.ndim!=1 or rr.shape!=aa.shape or rr.size<20:return 0.,'NO_LIDAR'
        diff=np.abs((aa-direction_local+math.pi)%(2*math.pi)-math.pi)
        if carrying and cargo_bearing is not None:
            projected=direction_dependent_extent(
                movement_relative_yaw=direction_local,
                cargo_bearing_body=cargo_bearing,
                uncertainty_rad=cargo_uncertainty,
                drone_radius_px=self.cfg.empty_radius_px)
            side=projected.lateral_px
            forward=projected.along_px
        else:
            side=(self.cfg.loaded_radius_px if carrying else self.cfg.empty_radius_px)
            forward=side
        # Direction-specific swept reach, not one full-circle cargo radius.
        selected=(diff<.45)&(np.abs(rr*np.sin(diff)) < side+4.)&np.isfinite(rr)
        if (semantic_target is not None and semantic_target.kind=='center'
                and near_target and semantic_target.distance<55.
                and abs(wrap(direction_local-semantic_target.angle))<.16):
            # The disposal center itself is the intended physical contact.
            # Permit only a slow, aligned approach when no OTHER return is
            # materially closer than the live semantic center surface.
            other=selected & (rr<float(semantic_target.distance)-12.)
            if not np.any(other):
                return min(float(desired_speed),.85),'DOCK_CONTACT'
        if semantic_target is not None and near_target:
            angle=float(semantic_target.angle);distance=float(semantic_target.distance)
            near_angle=np.abs((aa-angle+math.pi)%(2*math.pi)-math.pi)<math.radians(7.)
            # Ignore only a few rays whose range agrees with the physical target.
            # Do not erase other walls in a broad ±40-degree sector.
            selected &= ~(near_angle & (np.abs(rr-distance)<13.))
        if not np.any(selected):return 0.,'UNOBSERVED_DIRECTION'
        gap=float(np.min(rr[selected]))-forward-self.cfg.current_collision_margin_px
        if gap<0.5:return 0.,'INSUFFICIENT_CLEARANCE'
        allowed=min(float(desired_speed),self.braking_speed(gap,carrying))
        return allowed,'OK'

    def _choose_safe_velocity(self,pose:Pose,desired_world,rr,aa,
                              *,carrying,peer_relative=(),semantic_target=None,
                              near_target=False,cargo_bearing=None,
                        cargo_uncertainty=math.pi):
        v=np.asarray(desired_world,dtype=float).reshape(2)
        desired_speed=norm(v)
        if desired_speed<.03:return np.zeros(2),'HOLD'
        base=math.atan2(v[1],v[0])-pose.yaw
        # Discrete local velocity sampling inspired by DWA; every candidate
        # remains subject to the existing current LiDAR geometry check.
        candidates=[]
        for deviation in (0., math.radians(24.), -math.radians(24.),
                          math.radians(50.),-math.radians(50.),
                          math.radians(85.),-math.radians(85.),math.pi):
            ang=wrap(base+deviation)
            safe,reason=self._safe_direction(ang,desired_speed,rr,aa,
                  carrying=carrying,semantic_target=semantic_target,near_target=near_target,
                  cargo_bearing=cargo_bearing,cargo_uncertainty=cargo_uncertainty)
            if safe<.15:continue
            world=rotate((safe,0),pose.yaw+ang)
            conflict=0.
            for dist,bearing,peer_carry in peer_relative:
                if not math.isfinite(dist) or dist>115:continue
                toward=math.cos(wrap(ang-bearing))
                if toward<=.1:continue
                safety_radius=(62 if carrying else 52)
                if dist<safety_radius+safe*3.0:
                    # Carriers have right of way; peers are detected by sensors,
                    # not assumed visible through disabled radio links.
                    conflict += (25. if (peer_carry or not carrying) else 8.)*toward
            priority=3.6*abs(wrap(deviation)) + 1.8*(desired_speed-safe)+conflict
            if deviation==math.pi:priority+=7.0
            candidates.append((priority,safe,world,reason))
        if not candidates:
            return np.zeros(2),'ALL_CANDIDATES_BLOCKED'
        chosen=min(candidates,key=lambda c:(c[0],-c[1]))
        return chosen[2],('AVOID' if chosen[0]>.5 else chosen[3])

    def output(self,pose:Pose,measured_local_velocity,desired_local_velocity,
               *,lidar_ranges,lidar_angles,carrying=False,grasper=False,
               desired_yaw=None,peers=(),semantic_target=None,near_target=False,
               cargo_bearing=None,cargo_uncertainty=math.pi):
        v=np.asarray(desired_local_velocity,dtype=float).reshape(2)
        raw_speed=norm(v)
        ceiling=(self.cfg.loaded_speed_px_frame if carrying else self.cfg.empty_speed_px_frame)
        if raw_speed>ceiling:v=v*ceiling/raw_speed
        speed_cap=norm(v)
        safe,reason=self._choose_safe_velocity(pose,v,lidar_ranges,lidar_angles,
               carrying=carrying,peer_relative=peers,semantic_target=semantic_target,
               near_target=near_target,cargo_bearing=cargo_bearing,
               cargo_uncertainty=cargo_uncertainty)
        observed=np.asarray(measured_local_velocity,dtype=float).reshape(2)
        if not np.isfinite(observed).all():observed=np.zeros(2)
        if carrying:
            # Bounded empirical controller: no invented laden b coefficient.
            world_u=safe/max(1.0,self.cfg.loaded_speed_px_frame)+.19*(safe-observed)
        else:
            a=.95;b=.5729312289729012;p=.75
            world_u=(1.-a)/b*safe+(a-p)/b*(safe-observed)
        world_u=clip_disk(world_u)
        body=to_body(world_u,pose.yaw)
        yaw_error=0. if desired_yaw is None else wrap(float(desired_yaw)-pose.yaw)
        rotation=float(np.clip(.75*yaw_error,-.65,.65))
        if safe_speed:=norm(safe):
            if reason.startswith('ALL_CANDIDATES'):
                rotation=0.
        self.last_control=world_u
        self.last_obstacle_reason=reason
        return MotionResult(float(body[0]),float(body[1]),rotation,
                            1 if grasper else 0,reason,float(raw_speed),float(norm(safe)))


def make_reference(pos,waypoint,*,speed_cap,brake_decel,tolerance=9.):
    d=np.asarray(waypoint,dtype=float)-np.asarray(pos,dtype=float)
    length=norm(d)
    if length<=tolerance:return np.zeros(2)
    s=min(float(speed_cap), math.sqrt(max(0.,2.*brake_decel*(length-tolerance))))
    return normalize(d)*s

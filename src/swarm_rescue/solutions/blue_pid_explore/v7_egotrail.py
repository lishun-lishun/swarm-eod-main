"""GPS-independent SHORT odometry trail for a carrier's local retreat.

All poses live in an arbitrary ego-local frame integrated from *measured*
odometer increments; it is intentionally never aligned to simulation truth.
It can suggest a short backtrack toward recently visited free space, NOT
certify a full home route nor infer the disposal center location.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import math


def wrap(a: float) -> float:
    return (float(a)+math.pi)%(2*math.pi)-math.pi


@dataclass(frozen=True)
class EgoPose:
    x: float
    y: float
    heading: float
    step: int


@dataclass(frozen=True)
class RetreatRequest:
    bearing_body: float
    distance_px: float
    age_frames: int
    target: tuple[float,float]
    estimated_drift_px: float
    reason: str


class LocalOdomTrail:
    def __init__(self, *, max_history: int=450, min_record_px: float=4.0,
                 min_age_frames: int=8, distance_std_px: float=0.45,
                 heading_std_rad: float=math.radians(1.3)):
        self.max_history=int(max_history)
        self.min_record_px=float(min_record_px)
        self.min_age=int(min_age_frames)
        self.distance_std=float(distance_std_px)
        self.heading_std=float(heading_std_rad)
        self.reset()

    def reset(self):
        self.pose=EgoPose(0.,0.,0.,-1)
        self.history=deque(maxlen=self.max_history)
        self.valid=False
        self.gps_invalid_since=None
        self.last_odom_step=-1
        self.steering_suppressed_until=-1

    def update(self, *, step: int, odometry, gps_valid: bool) -> bool:
        """odom=(distance, travel_angle_from_previous_body, heading_delta).

        Must call once per control frame AFTER reading its legitimate odometer;
        it describes movement during the previous control interval.
        """
        if step<=self.last_odom_step:
            return False
        if odometry is None or len(odometry)!=3:
            return False
        d, alpha, theta=map(float,odometry)
        if not all(map(math.isfinite,(d,alpha,theta))):
            return False
        p=self.pose
        x=p.x+d*math.cos(p.heading+alpha)
        y=p.y+d*math.sin(p.heading+alpha)
        self.pose=EgoPose(x,y,wrap(p.heading+theta),int(step))
        self.last_odom_step=int(step)
        self.valid=True
        if gps_valid:
            self.gps_invalid_since=None
        elif self.gps_invalid_since is None:
            self.gps_invalid_since=int(step)
        if not self.history or math.dist((x,y),(self.history[-1].x,self.history[-1].y))>=self.min_record_px or step-self.history[-1].step>=self.min_age:
            self.history.append(self.pose)
        return True

    def retreat_target(self, *, now: int, path_missing: bool, carrying: bool,
                       local_segment_safe, min_distance_px=9.0,
                       max_distance_px=28.0, max_age_frames=135,
                       max_predicted_drift_px=15.0) -> RetreatRequest | None:
        """Suggest nearest preceding odometry waypoint for LOCAL exit only.

        Requires a live current-relative safety check. Calls to the legacy
        collision safety/controller still apply. On no route, caller must not
        substitute arbitrary world coordinates or assume full map connectivity.
        """
        if (not self.valid or not carrying or not path_missing
                or self.gps_invalid_since is None
                or now < self.steering_suppressed_until):
            return None
        cur=self.pose
        # Returning through recent *empty* traversal can fail when carrying;
        # each short edge must be reverified by current LiDAR/payload geometry.
        for old in reversed(self.history):
            age=max(0,int(now)-old.step)
            if age==0 or age>max_age_frames:
                continue
            dx,dy=old.x-cur.x,old.y-cur.y
            distance=math.hypot(dx,dy)
            if distance<min_distance_px:
                continue
            # Limit to the first reachable nearby breadcrumb, not a far point
            # that would skip an unseen wall corner.
            if distance>max_distance_px:
                break
            drift=math.sqrt(age)*self.distance_std + distance*self.heading_std*math.sqrt(age)
            if drift>max_predicted_drift_px:
                return None
            relative=wrap(math.atan2(dy,dx)-cur.heading)
            if local_segment_safe is None or not local_segment_safe(relative,distance):
                return None
            return RetreatRequest(relative,distance,age,(old.x,old.y),drift,
                                  'SHORT_REVERSE_ODOMETRY_SEGMENT')
        return None

    def suppress(self, now:int, frames:int=40):
        self.steering_suppressed_until=max(self.steering_suppressed_until,int(now)+int(frames))


def retreat_desired_world_velocity(request:RetreatRequest, current_world_heading:float,
                                   speed_px_per_frame:float=1.3)->tuple[float,float]:
    """Convert ego-local bearing into the existing controller's world-frame
    reference velocity; the world heading is the controller's LEGAL estimate.
    """
    bearing=float(current_world_heading)+request.bearing_body
    return (float(speed_px_per_frame)*math.cos(bearing),
            float(speed_px_per_frame)*math.sin(bearing))

"""V6: stateful path-integrity diagnostics, not a second obstacle controller.

Never triggers an unverified backoff or changes thrust. The caller reacts
transactionally: only replace a route after a verifiable new path exists.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RouteIssue:
    kind: str
    reason: str
    displacement_px: float = 0.0
    waypoint_progress_px: float = 0.0


class RouteIntegrityMonitor:
    def __init__(self, *, window_frames=65, progress_threshold_px=8.0,
                 min_translation_px=5.0, cooldown_frames=80):
        self.window=int(window_frames)
        self.min_progress=float(progress_threshold_px)
        self.min_translation=float(min_translation_px)
        self.cooldown=int(cooldown_frames)
        self.tag=None
        self.waypoint=None
        self.sample=deque(maxlen=self.window+1)
        self.last_issue=-10**8

    def reset(self):
        self.tag=None;self.waypoint=None;self.sample.clear();self.last_issue=-10**8

    def observe(self, *, step, route_tag, pose, waypoint, route_count,
                commanded_speed, center_cost_px=None,
                center_surface_distance_px=None, carrying=False,
                holding_for_traffic=False):
        if self.tag!=route_tag:
            self.sample.clear();self.tag=route_tag;self.waypoint=None
        p=(float(pose[0]),float(pose[1]))
        if not all(map(math.isfinite,p)):
            return RouteIssue('BAD_POSE','nonfinite legal estimate')
        if (carrying and center_cost_px is not None and math.isfinite(center_cost_px)
                and center_cost_px < 5.0 and center_surface_distance_px is not None
                and math.isfinite(center_surface_distance_px)
                and center_surface_distance_px > 85.0):
            return RouteIssue('WITNESS_NOT_CONTACT',
                              'center field zero yet center surface still distant')
        if route_count<=0 or waypoint is None:
            self.sample.clear();self.waypoint=None
            return RouteIssue('MISSING_ROUTE','no executable waypoint (not proof of arrival)')
        if holding_for_traffic:
            self.sample.clear()
            return RouteIssue('YIELD','intentional traffic wait')
        wp=tuple(float(v) for v in waypoint)
        if not all(map(math.isfinite,wp)):
            self.sample.clear();self.waypoint=None
            return RouteIssue('BAD_WAYPOINT','nonfinite route point')
        if self.waypoint is None or math.dist(self.waypoint,wp)>8.0:
            self.sample.clear();self.waypoint=wp
        d=math.dist(p,wp)
        if self.sample and step!=self.sample[-1][0]+1:
            self.sample.clear()
        self.sample.append((int(step),p,d))
        if len(self.sample)<2 or step-self.sample[0][0]<self.window:
            return RouteIssue('OK','short observation window')
        start=self.sample[0]
        motion=math.dist(start[1],p)
        advance=start[2]-d
        if (motion>=self.min_translation and advance>=self.min_progress):
            self.sample.clear();self.sample.append((int(step),p,d))
            return RouteIssue('OK','waypoint progress verified',motion,advance)
        if commanded_speed<=0.4:
            return RouteIssue('LOW_COMMAND','speed reference near zero',motion,advance)
        if step-self.last_issue<self.cooldown:
            return RouteIssue('COOLDOWN','recovery requested recently',motion,advance)
        self.last_issue=int(step)
        self.sample.clear();self.sample.append((int(step),p,d))
        return RouteIssue('STALLED','nonzero drive command without waypoint progress',motion,advance)

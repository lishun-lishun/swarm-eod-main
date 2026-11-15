"""Single finite-state mission supervisor: explore, pickup, carry, terminal.

Never replaces success with grab count or map-path completion.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .geometry import RelativeDetection, Pose, wrap, normalize
from .control import make_reference

@dataclass(frozen=True)
class MissionIntent:
    phase: str
    goal_xy: tuple[float,float]|None
    speed_cap: float
    grasp: bool
    terminal_target: RelativeDetection|None=None
    yaw: float|None=None
    note: str=''

class MissionSupervisor:
    def __init__(self):
        self.phase='EXPLORE'
        self.last_phase='BOOT'
        self.cargo_bearing=0.
        self.terminal_since=-1
        self.terminal_retries=0
        self.home_witness=None
        self.last_grasp=False
        self.last_goal=None
        self.last_step=-1
        self.pickup_since=-1

    def step(self,*,now:int,pose:Pose,carrying:bool,
             visible_bombs=(),visible_centers=(),task_xy=None,
             remembered_center_xy=None,explore_goal_xy=None,
             available_clearance=100.):
        if carrying and not self.last_grasp:
            self.pickup_since=now
            self.terminal_retries=0
            self.terminal_since=-1
            self.phase='CARRY'
        if not carrying and self.last_grasp:
            # Do not infer successful disposal here. TaskMemory does that.
            self.terminal_since=-1
            self.phase='EXPLORE'
        self.last_grasp=bool(carrying)
        self.last_step=now
        center=min(visible_centers,key=lambda z:z.distance) if visible_centers else None
        if carrying:
            # Live relative detection always outranks uncertain map center.
            if center is not None and center.distance<115.:
                self.phase='DOCK'
                if self.terminal_since<0:self.terminal_since=now
                direction=pose.yaw+center.angle
                distance=center.distance
                goal=(pose.x+max(0.,distance-3.)*math.cos(direction),
                      pose.y+max(0.,distance-3.)*math.sin(direction))
                # Final terminal touch is contact-driven; forward movement is
                # allowed toward the observed disposal surface, not through walls.
                cap=2.7 if distance>43 else 1.25
                yaw=wrap(direction-self.cargo_bearing) if distance<50. else None
                if distance<30 and now-self.terminal_since>48:
                    # Small bounded rotation to sweep the pivoted bomb across
                    # the disposal surface while continuing to hold grasper=1.
                    sign=1 if self.terminal_retries%2==0 else -1
                    yaw=wrap(pose.yaw+sign*.24)
                    if now-self.terminal_since>105:
                        self.terminal_retries+=1
                        self.terminal_since=now
                return MissionIntent('DOCK',goal,cap,True,center,yaw,'REAL_CENTER_SURFACE')
            self.phase='CARRY'
            if remembered_center_xy is not None:
                return MissionIntent('CARRY',tuple(remembered_center_xy),2.8,True,None,None,'KNOWN_CENTER_WITNESS')
            # If no center was ever seen, explore while carrying and KEEP GRASP.
            return MissionIntent('CARRY_EXPLORE',explore_goal_xy,2.0,True,None,None,'FIND_CENTER')
        if visible_bombs:
            closest=min((d for d in visible_bombs if not d.grasped),key=lambda z:z.distance,default=None)
            if closest is not None:
                self.phase='FETCH'
                bearing=pose.yaw+closest.angle
                # Semantic distance to surface; aim just behind it to grasp.
                target=(pose.x+(closest.distance+11.)*math.cos(bearing),
                        pose.y+(closest.distance+11.)*math.sin(bearing))
                self.cargo_bearing=closest.angle
                return MissionIntent('FETCH',target,3.4,True,closest,bearing,'LIVE_BOMB')
        if task_xy is not None:
            self.phase='FETCH'
            return MissionIntent('FETCH',tuple(task_xy),4.7,True,None,None,'KNOWN_BOMB')
        self.phase='EXPLORE'
        return MissionIntent('EXPLORE',explore_goal_xy,5.5,False,None,None,'FRONTIER')

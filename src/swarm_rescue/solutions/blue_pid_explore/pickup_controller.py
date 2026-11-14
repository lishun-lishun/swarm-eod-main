"""Finite-retry bomb pickup state machine using explicit grasp confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class PickupState(str, Enum):
    APPROACH_BOMB="APPROACH_BOMB"; ALIGN="ALIGN"; SLOW_APPROACH="SLOW_APPROACH"
    GRASP="GRASP"; CONFIRM="CONFIRM"; SUCCESS="SUCCESS"; FAILED="FAILED"


@dataclass(frozen=True)
class PickupCommand:
    state: PickupState
    target_velocity: np.ndarray
    grasper: int
    replan_required: bool


class PickupController:
    def __init__(self, *, approach_radius=35.0, slow_radius=14.0, grasp_radius=8.0,
                 max_retries=4):
        self.approach_radius=float(approach_radius); self.slow_radius=float(slow_radius)
        self.grasp_radius=float(grasp_radius); self.max_retries=int(max_retries); self.reset()

    def reset(self) -> None:
        self.state=PickupState.APPROACH_BOMB; self.retries=0; self.confirm_frames=0

    def compute(self, position: Sequence[float], velocity: Sequence[float], bomb_position: Sequence[float],
                *, grasped: bool) -> PickupCommand:
        if grasped:
            self.state=PickupState.SUCCESS
            return PickupCommand(self.state,np.zeros(2),1,False)
        p=np.asarray(position,float); v=np.asarray(velocity,float); target=np.asarray(bomb_position,float)
        delta=target-p; distance=float(np.linalg.norm(delta)); direction=delta/max(distance,1e-9)
        if self.state==PickupState.APPROACH_BOMB and distance<=self.approach_radius:
            self.state=PickupState.ALIGN
        if self.state==PickupState.ALIGN and np.linalg.norm(v)<=0.8:
            self.state=PickupState.SLOW_APPROACH
        if self.state==PickupState.SLOW_APPROACH and distance<=self.grasp_radius:
            self.state=PickupState.GRASP
        if self.state==PickupState.GRASP:
            self.state=PickupState.CONFIRM; self.confirm_frames=0
            return PickupCommand(PickupState.GRASP,np.zeros(2),1,False)
        if self.state==PickupState.CONFIRM:
            self.confirm_frames+=1
            if self.confirm_frames>8:
                self.retries+=1
                if self.retries>=self.max_retries:
                    self.state=PickupState.FAILED
                else:
                    self.state=PickupState.ALIGN
            return PickupCommand(self.state,-0.35*direction,1,self.state==PickupState.FAILED)
        speed=4.0 if self.state==PickupState.APPROACH_BOMB else (1.8 if self.state==PickupState.ALIGN else 0.7)
        return PickupCommand(self.state,speed*direction,1 if distance<=self.slow_radius else 0,
                             self.state==PickupState.FAILED)


